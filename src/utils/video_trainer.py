import os
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, CenterCrop, RandomErasing
from collections import defaultdict

from models.avmnet import AVMNet
from dataset.metric_learning_dataset import MetricLearningDataset
from dataset.samplers import RandomSampler
from dataset.transforms import ToTensorVideo, NormalizeVideo
from utils.utils import *


class VideoTrainer():
    def __init__(self, config):
        self.config = config

        # Config
        self.set_parameters_by_config()

        # Datasets & DataLoader
        print('\n--- Dataset ---')
        self.load_dataset()

        # Model
        print('\n--- Model ---')
        self.load_model()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay']
        )
        # self.scheduler = CosineScheduler(config['optimizer']['lr'], config['train']['n_epochs'])
        # self.scheduler = CosineAnnealingLR(
        #                     self.optimizer,
        #                     T_max=config['train']['n_epochs'],
        #                     eta_min=config['optimizer']['lr']/100,)
        self.scaler = amp.GradScaler()

        # Move to GPU
        self.model = torch.nn.DataParallel(self.model, device_ids=config['train']['gpu_ids']).to(self.device)

        # # Move optimizer to GPU
        # for param in self.optimizer.state.values():
        #     # Not sure there are any global tensors in the state dict
        #     if isinstance(param, torch.Tensor):
        #         param.data = param.data.to(self.device)
        #         if param._grad is not None:
        #             param._grad.data = param._grad.data.to(self.device)
        #     elif isinstance(param, dict):
        #         for subparam in param.values():
        #             if isinstance(subparam, torch.Tensor):
        #                 subparam.data = subparam.data.to(self.device)
        #                 if subparam._grad is not None:
        #                     subparam._grad.data = subparam._grad.data.to(self.device)

        # Tensorboard
        if config['tensorboard']['comment']:
            self.writer = SummaryWriter(
                comment=f"{config['tensorboard']['comment']}")
        else:
            self.writer = None

        # Check if save dir exists
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        print('\n--- Start Training ---')
        for ep in range(self.current_ep+1, self.n_eps):
            print(f'\nEpoch {ep}')
            self.current_ep = ep
            self.train_one_epoch()
            if self.use_teacher:
                self.model.module.update_teacher()
            # self.scheduler.step()
            if ep % self.save_frequency == 0:
                print('Validation:')
                self.validate()
            if self.patience > 0 and self.end_cycle >= self.patience:
                break

    def train_one_epoch(self):
        loop = tqdm(
            self.train_loader,
            leave=True,
            desc=f'Train Epoch:{self.current_ep}/{self.n_eps}'
        )

        ep_loss = defaultdict(int)

        if self.show_metric:
            video_to_logits = defaultdict(list)
            video_to_labels = {}

        self.model.train()
        for it, data in enumerate(loop):
            vid, aud, label, vpath = data

            p = float(it + self.current_ep * len(self.train_loader)) / self.n_eps / len(self.train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            vid = vid.to(self.device, dtype=torch.float)
            aud = aud.to(self.device, dtype=torch.float)

            label = label.to(self.device)

            total_loss = 0

            with amp.autocast():
                output = self.model(vid, aud, alpha=alpha, y=label)

            # feature learning
            if 'NTXent' in self.use_losses:
                if self.model.module.setting == 'AVTS':
                    nce_AVTS_loss = output['NTXent_AVTS'].mean()
                    total_loss += nce_AVTS_loss
                    ep_loss['NTXent_AVTS'] += nce_AVTS_loss.item()
                elif self.model.module.setting == 'AVC':
                    nce_AVC_loss = output['NTXent_AVC'].mean()
                    total_loss += nce_AVC_loss
                    ep_loss['NTXent_AVC'] += nce_AVC_loss.item()
                elif self.model.module.setting == 'AVM':
                    nce_AVTS_loss = output['NTXent_AVTS'].mean()
                    nce_AVC_loss = output['NTXent_AVC'].mean()
                    total_loss += nce_AVTS_loss*10 + nce_AVC_loss
                    ep_loss['NTXent_AVTS'] += nce_AVTS_loss.item()
                    ep_loss['NTXent_AVC'] += nce_AVC_loss.item()

            if 'L2' in self.use_losses:
                ema_loss = output['EMA'].mean()
                total_loss += ema_loss
                ep_loss['EMA'] += ema_loss.item()

            # supervised learning
            if 'BCE' in self.use_losses:
                bce_loss = output['BCE'].mean()
                total_loss += bce_loss
                ep_loss['BCE'] += bce_loss.item()

            # domain loss
            if 'Adversarial' in self.use_losses:
                domain_loss = output['Adversarial'].mean()
                total_loss += domain_loss
                ep_loss['Domain'] += domain_loss.item()

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # total_loss.backward()
            # self.optimizer.step()

            if self.show_metric:
                for i in range(len(vpath)):
                    if 'FakeAVCeleb' in vpath[i]:
                        video_id = '_'.join(vpath[i].split('/')[-6:])
                    elif 'FaceForensics' in vpath[i]:
                        video_id = '_'.join(vpath[i].split('/')[-4:])
                    elif 'LRW' in vpath[i]:
                        video_id = vpath[i].split('/')[-1]

                    if self.augmented_type == 'AVTS':
                        video_id += '_' + str(int(label[i].item()))

                    video_to_logits[video_id].append(output['logits'][i].view(-1).detach().cpu())
                    video_to_labels[video_id] = label[i].view(-1).detach().cpu()

            loop.set_postfix(
                # alpha=alpha if 'Adversarial' in self.use_losses else 0,
                # da=domain_loss.item() if 'Adversarial' in self.use_losses else 0,
                AVTS=nce_AVTS_loss.item() if self.model.module.setting in ['AVTS', 'AVM'] else 0,
                AVC=nce_AVC_loss.item() if self.model.module.setting in ['AVC', 'AVM'] else 0,
                EMA=ema_loss.item() if 'L2' in self.use_losses else 0,
                # bce=bce_loss.item(),
            )

        if self.show_metric:
            auc_video = compute_video_level_auc(video_to_logits, video_to_labels)
            acc_video = compute_video_level_acc(video_to_logits, video_to_labels)
            auc_clip = compute_clip_level_auc(video_to_logits, video_to_labels)
            acc_clip = compute_clip_level_acc(video_to_logits, video_to_labels)

        print(f'lr rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
        for loss_type in ep_loss:
            ep_loss[loss_type] /= len(self.train_loader)
            print(f'ep {loss_type} loss: {ep_loss[loss_type]:.7f}', end=' ')
        if self.show_metric:
            print(
                f'\nclip auc: {auc_clip:.3f}, acc: {acc_clip:.3f}, video auc: {auc_video:.3f}, acc: {acc_video:.3f}')
        else:
            print()
        if self.writer:
            for loss_type in ep_loss:
                self.writer.add_scalar(f'{loss_type}_Loss/train', ep_loss[loss_type], self.current_ep)
            if self.show_metric:
                self.writer.add_scalar('Accuracy_clip/train',
                                    acc_clip, self.current_ep)
                self.writer.add_scalar(
                    'AUC_Score_clip/train', auc_clip, self.current_ep)
                self.writer.add_scalar(
                    'Accuracy_video/train', acc_video, self.current_ep)
                self.writer.add_scalar(
                    'AUC_Score_video/train', auc_video, self.current_ep)

        return

    def validate(self):
        ep_loss = defaultdict(int)

        if self.show_metric:
            video_to_logits = defaultdict(list)
            video_to_labels = {}

        self.model.eval()
        with torch.no_grad():
            for it, data in enumerate(tqdm(self.val_loader)):
                vid, aud, label, vpath = data

                vid = vid.to(self.device, dtype=torch.float)
                aud = aud.to(self.device, dtype=torch.float)
                label = label.to(self.device)

                with amp.autocast():
                    output = self.model(vid, aud, alpha=0, y=label)

                # feature learning
                if 'NTXent' in self.use_losses:
                    if self.model.module.setting == 'AVTS' or self.model.module.setting == 'AVM':
                        ep_loss['NTXent_AVTS'] += output['NTXent_AVTS'].mean().item()
                    if self.model.module.setting == 'AVC' or self.model.module.setting == 'AVM':
                        ep_loss['NTXent_AVC'] += output['NTXent_AVC'].mean().item()

                if 'L2' in self.use_losses:
                    ep_loss['L2'] += output['EMA'].mean().item()

                # supervised learning
                if 'BCE' in self.use_losses:
                    loss = output['BCE'].mean()
                    ep_loss['BCE'] += loss.item()

                if self.show_metric:
                    for i in range(len(vpath)):
                        if 'FakeAVCeleb' in vpath[i]:
                            video_id = '_'.join(vpath[i].split('/')[-6:])
                        elif 'FaceForensics' in vpath[i]:
                            video_id = '_'.join(vpath[i].split('/')[-4:])
                        elif 'LRW' in vpath[i]:
                            video_id = vpath[i].split('/')[-1]

                        if self.augmented_type == 'AVTS':
                            video_id += '_' + str(int(label[i].item()))

                        video_to_logits[video_id].append(output['logits'][i].view(-1).detach().cpu())
                        video_to_labels[video_id] = label[i].view(-1).detach().cpu()

        if self.show_metric:
            auc_video = compute_video_level_auc(video_to_logits, video_to_labels)
            acc_video = compute_video_level_acc(video_to_logits, video_to_labels)
            auc_clip = compute_clip_level_auc(video_to_logits, video_to_labels)
            acc_clip = compute_clip_level_acc(video_to_logits, video_to_labels)
        for loss_type in ep_loss:
            ep_loss[loss_type] /= len(self.val_loader)
            print(f'ep {loss_type} loss: {ep_loss[loss_type]:.7f}', end=' ')
        if self.show_metric:
            print(
                f'\nclip auc: {auc_clip:.3f}, acc: {acc_clip:.3f}, video auc: {auc_video:.3f}, acc: {acc_video:.3f}')
        else:
            print()
        print(f'last save ep: {self.last_save_ep}')

        if self.writer:
            for loss_type in ep_loss:
                self.writer.add_scalar(f'{loss_type}_Loss/val', ep_loss[loss_type], self.current_ep)
            if self.show_metric:
                self.writer.add_scalar('Accuracy_clip/val',
                                    acc_clip, self.current_ep)
                self.writer.add_scalar('AUC_Score_clip/val',
                                    auc_clip, self.current_ep)
                self.writer.add_scalar('Accuracy_video/val',
                                    acc_video, self.current_ep)
                self.writer.add_scalar('AUC_Score_video/val',
                                    auc_video, self.current_ep)

        if 'loss' in self.val_policy:
            loss = 0
            if self.model.module.setting == 'AVTS' or self.model.module.setting == 'AVM':
                loss += ep_loss['NTXent_AVTS']
            if self.model.module.setting == 'AVC' or self.model.module.setting == 'AVM':
                loss += ep_loss['NTXent_AVC']
            if 'L2' in self.use_losses:
                loss += ep_loss['L2']

            if loss < self.current_val_loss:
                self.current_val_loss = loss
                self.last_save_ep = self.current_ep
                self.save_checkpoint('best_loss')
                self.end_cycle = -1
        if 'auc' in self.val_policy:
            if auc_video > self.current_val_auc:
                self.current_val_auc = auc_video
                self.last_save_ep = self.current_ep
                self.save_checkpoint('best_auc')
                self.end_cycle = -1
        self.end_cycle += 1

        return

    def save_checkpoint(self, ckp_name):
        checkpoint = {
            'model': self.model.module.state_dict() if len(self.gpu_ids) > 1 else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'scheduler': self.scheduler.state_dict(),
            'current_val_loss': self.current_val_loss,
            'current_ep': self.current_ep,
        }
        checkpoint_path = os.path.join(self.save_dir, f'{ckp_name}.pth')
        torch.save(checkpoint, checkpoint_path)
        print("$ Save checkpoint to '{}'".format(checkpoint_path))
        return

    def resume(self, ckp_path, only_model=True):
        resume_parameters = 0
        checkpoint = torch.load(ckp_path, map_location='cpu')
        # self.model.load_state_dict(checkpoint['model'])
        for name, param in checkpoint['model'].items():
            if name in self.model.state_dict():
                self.model.state_dict()[name].copy_(param)
                resume_parameters += param.numel()
        if not only_model:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'current_loss' in checkpoint:
                self.current_val_loss = checkpoint['current_loss']
            else:
                self.current_val_loss = checkpoint['current_val_loss']
            self.current_ep = checkpoint['current_ep']

        print(f'$ Resume {resume_parameters/1_000_000:.3f}M Parameters From {ckp_path}')        

    def set_parameters_by_config(self):
        self.device = torch.device(
            f"cuda:{self.config['train']['gpu_ids'][0]}" if torch.cuda.is_available() else 'cpu')
        self.gpu_ids = self.config['train']['gpu_ids']
        self.n_eps = self.config['train']['n_epochs']
        self.use_losses = self.config['train']['loss']
        self.save_dir = self.config['train']['save_dir']
        self.save_frequency = self.config['train']['save_frequency']
        self.last_save_ep = -1
        self.current_ep = -1
        self.current_val_loss = float('inf')
        self.current_val_auc = 0
        self.val_policy = self.config['train']['val_policy']
        self.patience = self.config['train']['patience']
        self.end_cycle = 0
        self.mode = self.config['train']['mode']
        self.orig_lr = self.config['optimizer']['lr']
        self.augmented_type = self.config['dataset']['augmented_type']
        self.shift_type = self.config['dataset']['shift_type']
        self.show_metric = self.config['train']['show_metric']
        self.use_teacher = self.config['model']['use_teacher']

    def load_dataset(self):
        # Dataset
        print(f'$ Loading Train Dataset...')
        train_transform = {
            'video': Compose([
                        ToTensorVideo(),
                        RandomCrop((self.config['dataset']['size'], self.config['dataset']['size'])),
                        RandomHorizontalFlip(0.5),
                        RandomErasing(),
                        NormalizeVideo((0.421,), (0.165,))
                    ]),
            'audio': None,
        }
        self.train_dataset = MetricLearningDataset(
            root=self.config['dataset']['root'],
            transform=train_transform,
            mode='train',
            fps=self.config['dataset']['fps'],
            duration=self.config['dataset']['duration'],
            shift_type=self.config['dataset']['shift_type'],
            img_type=self.config['dataset']['img_type'],
            aud_feat=self.config['dataset']['aud_feat'],
            grayscale=self.config['dataset']['grayscale'],
        )

        print(f'$ Loading Validation Dataset...')
        val_transform = {
            'video': Compose([
                        ToTensorVideo(),
                        CenterCrop((self.config['dataset']['size'], self.config['dataset']['size'])),
                        NormalizeVideo((0.421,), (0.165,))
                    ]),
            'audio': None,
        }
        self.val_dataset = MetricLearningDataset(
            root=self.config['dataset']['root'],
            transform=val_transform,
            mode='val',
            fps=self.config['dataset']['fps'],
            duration=self.config['dataset']['duration'],
            shift_type=self.config['dataset']['shift_type'],
            img_type=self.config['dataset']['img_type'],
            aud_feat=self.config['dataset']['aud_feat'],
            grayscale=self.config['dataset']['grayscale'],
        )

        # Data Loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['dataloader']['batch_size'],
            num_workers=self.config['dataloader']['num_workers'],
            sampler=RandomSampler(self.train_dataset, len(self.val_dataset)*4),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['dataloader']['batch_size'],
            num_workers=self.config['dataloader']['num_workers'],
            shuffle=False
        )

        # Print Dataset Info
        print(f'Train Dataset Size: {len(self.train_dataset)}')
        print(f'Validation Dataset Size: {len(self.val_dataset)}')

    def load_model(self):
        # Model
        self.model = AVMNet(
            backbone=self.config['model']['backbone'],
            last_dim=self.config['model']['last_dim'],
            frames_per_clip=int(self.config['dataset']['fps'] * self.config['dataset']['duration']),
            predict_label=self.config['model']['predict_label'],
            use_losses=self.config['train']['loss'],
            aud_feat=self.config['dataset']['aud_feat'],
            setting=self.config['train']['setting'],
            img_in_dim=1 if self.config['dataset']['grayscale'] else 3,
            use_teacher=self.config['model']['use_teacher'],
        )

        # Resume parameters
        if config['train']['resume'] and os.path.isfile(config['train']['resume']):
            self.resume(config['train']['resume'], config['train']['only_model'])

        # Print Model Info
        print()
        max_model_name_length = max(map(len, config['model']['backbone']))
        print('-'*(max_model_name_length+40))

        vid_model_parameters = sum([p.numel() for p in self.model.v_encoder.parameters()])
        print(f'  Video   Model "{config["model"]["backbone"][0]:^{max_model_name_length}}" Parameters: {vid_model_parameters/1_000_000:.3f}M')

        aud_model_parameters = sum([p.numel() for p in self.model.a_encoder.parameters()])
        print(f'  Audio   Model "{config["model"]["backbone"][1]:^{max_model_name_length}}" Parameters: {aud_model_parameters/1_000_000:.3f}M')

        classify_model_parameters = 0
        print(f' Classify Model "{config["model"]["backbone"][2]:^{max_model_name_length}}" Parameters: {classify_model_parameters/1_000_000:.3f}M')
        print('-'*(max_model_name_length+40))

        total_model_parameters = sum([p.numel() for p in self.model.parameters()])
        print(f'  Total   Model {" "*(max_model_name_length+2)} Parameters: {total_model_parameters/1_000_000:.3f}M')

        trainable_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad == True])
        print(f'Trainable Model {" "*(max_model_name_length+2)} Parameters: {trainable_parameters/1_000_000:.3f}M')
        print('-'*(max_model_name_length+40))