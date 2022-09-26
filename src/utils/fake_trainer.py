from dataclasses import replace
import os
import torch
import numpy as np

from tqdm import tqdm
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, CenterCrop
from collections import defaultdict

from models.fakenet import FakeNet
from dataset.fake_dataset import FakeDataset
from dataset.FakeAVCeleb_dataset import FakeAVCelebDataset
from dataset.samplers import ImbalancedDatasetSampler
from utils.utils import *


class FakeTrainer():

    def __init__(self, config):
        self.config = config
        # Config
        self.set_parameters_by_config()

        # Datasets & Data loader
        print('\n--- Dataset ---')
        self.load_dataset()

        if config['model']['logit_adjustment']:
            self.logit_adjustment = self.compute_adjustment()
        else:
            self.logit_adjustment = None

        # Model
        print('\n--- Model ---')
        self.load_model()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay'],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            config['train']['n_epochs'],
            eta_min=config['optimizer']['lr'] / 100,
            last_epoch=-1)
        self.scaler = amp.GradScaler()

        # Move to GPU
        self.model = torch.nn.DataParallel(
            self.model, device_ids=config['train']['gpu_ids']).to(self.device)

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
        for ep in range(self.current_ep + 1, self.n_eps):
            print(f'\nEpoch {ep}')
            self.current_ep = ep
            self.train_one_epoch()
            self.scheduler.step()
            if ep % self.save_frequency == 0:
                print('Validation:')
                self.validate()
            if self.patience > 0 and self.end_cycle >= self.patience:
                break

    def train_one_epoch(self):
        loop = tqdm(self.train_loader,
                    leave=True,
                    desc=f'Train Epoch:{self.current_ep}/{self.n_eps}')

        ep_loss = defaultdict(int)

        if self.show_metric:
            video_to_logits = defaultdict(list)
            video_to_labels = {}

        self.model.train()
        for it, data in enumerate(loop):
            x, label, vpath, fake_type_label = data

            x = list(map(lambda x: x.to(self.device, dtype=torch.float), x))
            label = label.to(self.device)

            p = float(it + self.current_ep * len(self.train_loader)
                      ) / self.n_eps / len(self.train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            total_loss = 0

            with amp.autocast():
                output = self.model(x,
                                    y=label,
                                    alpha=alpha,
                                    fake_type_label=fake_type_label
                                    if self.predict_fake_label else None)

            # supervised learning
            if 'BCE' in self.use_losses:
                bce_loss = output['BCE'].mean()
                total_loss += bce_loss
                ep_loss['BCE'] += bce_loss.item() / len(self.train_loader)

            if self.predict_fake_label:
                fake_type_loss = output['FakeType'].mean()
                total_loss += fake_type_loss
                ep_loss['FakeType'] += fake_type_loss.item() / len(
                    self.train_loader)

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

                    video_to_logits[video_id].append(
                        output['logits'][i].view(-1).detach().cpu())
                    video_to_labels[video_id] = label[i].view(
                        -1).detach().cpu()

            loop.set_postfix(
                bce=bce_loss.item(),
                fake_da=fake_type_loss.item()
                if self.predict_fake_label else 0,
            )

        if self.show_metric:
            auc_video = compute_video_level_auc(video_to_logits,
                                                video_to_labels)
            acc_video = compute_video_level_acc(video_to_logits,
                                                video_to_labels)
            auc_clip = compute_clip_level_auc(video_to_logits, video_to_labels)
            acc_clip = compute_clip_level_acc(video_to_logits, video_to_labels)

        print(f'lr rate: {self.optimizer.param_groups[0]["lr"]:.8f}')
        for loss_type in ep_loss:
            print(f'ep {loss_type} loss: {ep_loss[loss_type]:.3f}', end=' ')
        if self.show_metric:
            print(
                f'\nClip auc: {auc_clip:.3f}, acc: {acc_clip:.3f}, Video auc: {auc_video:.3f}, acc: {acc_video:.3f}'
            )
        else:
            print()
        if self.writer:
            self.writer.add_scalar(
                'All_Loss/train',
                sum([ep_loss[t] for t in ep_loss]) / len(ep_loss),
                self.current_ep)
            if self.show_metric:
                self.writer.add_scalar('Accuracy_clip/train', acc_clip,
                                       self.current_ep)
                self.writer.add_scalar('AUC_Score_clip/train', auc_clip,
                                       self.current_ep)
                self.writer.add_scalar('Accuracy_video/train', acc_video,
                                       self.current_ep)
                self.writer.add_scalar('AUC_Score_video/train', auc_video,
                                       self.current_ep)

        return

    def validate(self):
        ep_loss = defaultdict(int)

        if self.show_metric:
            video_to_logits = defaultdict(list)
            video_to_labels = {}
            video_per_type_to_logits = defaultdict(lambda: defaultdict(list))
            video_per_type_to_labels = defaultdict(dict)

        self.model.eval()
        with torch.no_grad():
            for it, data in enumerate(tqdm(self.val_loader)):
                x, label, vpath, _ = data

                x = list(map(lambda x: x.to(self.device, dtype=torch.float),
                             x))
                label = label.to(self.device)

                with amp.autocast():
                    output = self.model(x, y=label)

                # supervised learning
                if 'BCE' in self.use_losses:
                    bce_loss = output['BCE'].mean()
                    ep_loss['BCE'] += bce_loss.item() / len(self.val_loader)

                if self.show_metric:
                    for i in range(len(vpath)):
                        if 'FakeAVCeleb' in vpath[i]:
                            video_id = '_'.join(vpath[i].split('/')[-6:])
                            video_type = vpath[i].split('/')[-6]
                        elif 'FaceForensics' in vpath[i]:
                            video_id = '_'.join(vpath[i].split('/')[-4:])
                            video_type = vpath[i].split('/')[-4]
                        elif 'LRW' in vpath[i]:
                            video_id = vpath[i].split('/')[-1]

                        video_to_logits[video_id].append(
                            output['logits'][i].view(-1).detach().cpu())
                        video_to_labels[video_id] = label[i].view(
                            -1).detach().cpu()
                        video_per_type_to_logits[video_type][video_id].append(
                            output['logits'][i].view(-1).detach().cpu())
                        video_per_type_to_labels[video_type][video_id] = label[
                            i].view(-1).detach().cpu()

        if self.show_metric:
            auc_video = compute_video_level_auc(video_to_logits,
                                                video_to_labels)
            acc_video = compute_video_level_acc(video_to_logits,
                                                video_to_labels)
            auc_clip = compute_clip_level_auc(video_to_logits, video_to_labels)
            acc_clip = compute_clip_level_acc(video_to_logits, video_to_labels)
        for loss_type in ep_loss:
            print(f'ep {loss_type} loss: {ep_loss[loss_type]:.3f}', end=' ')
        if self.show_metric:
            print(
                f'\nClip auc: {auc_clip:.3f}, acc: {acc_clip:.3f}, Video auc: {auc_video:.3f}, acc: {acc_video:.3f}'
            )
            for t in video_per_type_to_logits:
                acc = compute_video_level_acc(video_per_type_to_logits[t],
                                              video_per_type_to_labels[t])
                print(f'Type {t} acc: {acc:.3f}')
        else:
            print()
        print(f'Last Save ep: {self.last_save_ep}')
        if 'loss' in self.val_policy:
            print(f'Best loss: {self.current_val_loss:.3f}')
        if 'auc' in self.val_policy:
            print(f'Best auc: {self.current_val_auc:.3f}')

        if self.writer:
            self.writer.add_scalar(
                'All_Loss/val',
                sum([ep_loss[t] for t in ep_loss]) / len(ep_loss),
                self.current_ep)
            if self.show_metric:
                self.writer.add_scalar('Accuracy_clip/val', acc_clip,
                                       self.current_ep)
                self.writer.add_scalar('AUC_Score_clip/val', auc_clip,
                                       self.current_ep)
                self.writer.add_scalar('Accuracy_video/val', acc_video,
                                       self.current_ep)
                self.writer.add_scalar('AUC_Score_video/val', auc_video,
                                       self.current_ep)

        if 'loss' in self.val_policy:
            if ep_loss[self.use_losses[0]] < self.current_val_loss:
                self.current_val_loss = ep_loss[self.use_losses[0]]
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
            'model':
            self.model.module.state_dict()
            if len(self.gpu_ids) > 1 else self.model.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
            # 'scheduler': self.scheduler.state_dict(),
            # 'current_val_loss': self.current_val_loss,
            'current_ep':
            self.current_ep,
        }
        checkpoint_path = os.path.join(self.save_dir, f'{ckp_name}.pth')
        torch.save(checkpoint, checkpoint_path)
        print("$ Save Checkpoint To '{}'".format(checkpoint_path))
        return

    def resume(self, ckp_path, only_model=True):
        resume_parameters = 0
        checkpoint = torch.load(ckp_path, map_location='cpu')
        # self.model.load_state_dict(checkpoint['model'])
        for name, param in checkpoint['model'].items():
            name = name.lstrip('module.')
            if name in self.model.state_dict():
                # print(name)
                self.model.state_dict()[name].copy_(param)
                resume_parameters += param.numel()
        if not only_model:
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            # if 'current_loss' in checkpoint:
            #     self.current_val_loss = checkpoint['current_loss']
            # else:
            #     self.current_val_loss = checkpoint['current_val_loss']
            self.current_ep = checkpoint['current_ep']

        print(
            f'$ Resume {resume_parameters/1_000_000:.3f}M Parameters From {ckp_path}'
        )

    def set_parameters_by_config(self):
        self.device = torch.device(f"cuda:{self.config['train']['gpu_ids'][0]}"
                                   if torch.cuda.is_available() else 'cpu')
        self.gpu_ids = self.config['train']['gpu_ids']
        self.n_eps = self.config['train']['n_epochs']
        self.save_dir = self.config['train']['save_dir']
        self.save_frequency = self.config['train']['save_frequency']
        self.last_save_ep = -1
        self.use_losses = self.config['train']['loss']
        self.current_ep = -1
        self.current_val_loss = float('inf')
        self.current_val_auc = 0
        self.val_policy = self.config['train']['val_policy']
        self.patience = self.config['train']['patience']
        self.end_cycle = 0
        self.mode = self.config['train']['mode']
        self.orig_lr = self.config['optimizer']['lr']
        self.show_metric = self.config['train']['show_metric']
        self.predict_fake_label = self.config['model']['predict_fake_label']
        self.frames_per_clip = int(self.config['dataset']['fps'] *
                                   self.config['dataset']['duration'])

    def load_dataset(self):
        # Dataset
        print(f'$ Loading Train Dataset...')
        if 'FaceForensics' in self.config['dataset']['root']:
            self.train_dataset = FakeDataset(
                mode='train',
                input_mode=self.config['train']['mode'],
                root=self.config['dataset']['root'],
                fps=self.config['dataset']['fps'],
                duration=self.config['dataset']['duration'],
                img_size=self.config['dataset']['size'],
                img_type=self.config['dataset']['img_type'],
                grayscale=self.config['dataset']['grayscale'],
                aud_feat=self.config['dataset']['aud_feat'],
                fake_types=self.config['dataset']['train_fake_types'],
                use_percentage=self.config['dataset']['use_percentage'],
            )
        elif 'FakeAVCeleb' in self.config['dataset']['root']:
            self.train_dataset = FakeAVCelebDataset(
                mode='train',
                root=self.config['dataset']['root'],
                fps=self.config['dataset']['fps'],
                duration=self.config['dataset']['duration'],
                img_size=self.config['dataset']['size'],
                img_type=self.config['dataset']['img_type'],
                grayscale=self.config['dataset']['grayscale'],
                aud_feat=self.config['dataset']['aud_feat'],
            )
        else:
            raise NotImplementedError

        print(f'$ Loading Validation Dataset...')
        if 'FaceForensics' in self.config['dataset']['root']:
            self.val_dataset = FakeDataset(
                mode='val',
                input_mode=self.config['train']['mode'],
                root=self.config['dataset']['root'],
                fps=self.config['dataset']['fps'],
                duration=self.config['dataset']['duration'],
                img_size=self.config['dataset']['size'],
                img_type=self.config['dataset']['img_type'],
                grayscale=self.config['dataset']['grayscale'],
                aud_feat=self.config['dataset']['aud_feat'],
                fake_types=self.config['dataset']['test_fake_types'],
            )
        elif 'FakeAVCeleb' in self.config['dataset']['root']:
            self.val_dataset = FakeAVCelebDataset(
                mode='val',
                root=self.config['dataset']['root'],
                fps=self.config['dataset']['fps'],
                duration=self.config['dataset']['duration'],
                img_size=self.config['dataset']['size'],
                img_type=self.config['dataset']['img_type'],
                grayscale=self.config['dataset']['grayscale'],
                aud_feat=self.config['dataset']['aud_feat'],
            )
        else:
            raise NotImplementedError

        # Data loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['dataloader']['batch_size'],
            num_workers=self.config['dataloader']['num_workers'],
            sampler=ImbalancedDatasetSampler(
                self.train_dataset,
                num_samples=max(
                    len(self.train_dataset) // self.frames_per_clip,
                    self.config['dataloader']['batch_size'])),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['dataloader']['batch_size'],
            num_workers=self.config['dataloader']['num_workers'],
            shuffle=False,
        )

        # Print Dataset Info
        print(f'Train Dataset Size: {len(self.train_dataset)}')
        print('Dataset Type:')
        print(self.train_loader.sampler.label_to_count)
        print(f'Validation Dataset Size: {len(self.val_dataset)}')

    def load_model(self):
        # Model
        self.model = FakeNet(
            backbone=self.config['model']['backbone'],
            img_in_dim=1 if self.config['dataset']['grayscale'] else 3,
            last_dim=self.config['model']['last_dim'],
            frames_per_clip=int(self.config['dataset']['fps'] *
                                self.config['dataset']['duration']),
            mode=self.config['train']['mode'],
            predict_label=self.config['model']['predict_label'],
            use_losses=self.config['train']['loss'],
            aud_feat=self.config['dataset']['aud_feat'],
            normalized=self.config['model']['normalized'],
            logit_adjustment=self.logit_adjustment,
            fake_classes=len(self.config['dataset']['train_fake_types']),
            predict_fake_label=self.config['model']['predict_fake_label'],
            modality_embedding=self.config['model']['modality_embedding'],
        )

        # Resume Model
        if self.config['train']['resume']:
            if os.path.isfile(self.config['train']['resume']):
                self.resume(self.config['train']['resume'],
                            self.config['train']['only_model'])
            else:
                raise FileNotFoundError(
                    f"{self.config['train']['resume']} is not found")

        # Freeze Feature Extrator
        if self.config['train']['freeze_feature_extractor']:
            if 'V' in self.mode:
                print('$ Freeze video feature extractor')
                for param in self.model.v_encoder.parameters():
                    param.requires_grad = False
            if 'A' in self.mode:
                print('$ Freeze audio feature extractor')
                for param in self.model.a_encoder.parameters():
                    param.requires_grad = False

        # Print Model Info
        max_model_name_length = max(map(len, self.config['model']['backbone']))
        print('-' * (max_model_name_length + 40))

        vid_model_parameters = sum([
            p.numel() for p in self.model.v_encoder.parameters()
        ]) if 'V' in self.config['train']['mode'] else 0
        print(
            f'  Video   Model "{self.config["model"]["backbone"][0]:^{max_model_name_length}}" Parameters: {vid_model_parameters/1_000_000:.3f}M'
        )

        aud_model_parameters = sum([
            p.numel() for p in self.model.a_encoder.parameters()
        ]) if 'A' in self.config['train']['mode'] else 0
        print(
            f'  Audio   Model "{self.config["model"]["backbone"][1]:^{max_model_name_length}}" Parameters: {aud_model_parameters/1_000_000:.3f}M'
        )

        classify_model_parameters = sum(
            [p.numel() for p in self.model.temporal_classifier.parameters()])
        print(
            f' Classify Model "{self.config["model"]["backbone"][2]:^{max_model_name_length}}" Parameters: {classify_model_parameters/1_000_000:.3f}M'
        )
        print('-' * (max_model_name_length + 40))

        total_model_parameters = sum(
            [p.numel() for p in self.model.parameters()])
        print(
            f'  Total   Model {" "*(max_model_name_length+2)} Parameters: {total_model_parameters/1_000_000:.3f}M'
        )

        trainable_parameters = sum([
            p.numel() for p in self.model.parameters()
            if p.requires_grad == True
        ])
        print(
            f'Trainable Model {" "*(max_model_name_length+2)} Parameters: {trainable_parameters/1_000_000:.3f}M'
        )
        print('-' * (max_model_name_length + 40))

    def compute_adjustment(self, tro=1.0):
        """compute the base probabilities"""

        label_freq = {}
        for i, (inputs, target, _) in enumerate(self.train_loader):
            target = target.to(self.device)
            for j in target:
                key = int(j.item())
                label_freq[key] = label_freq.get(key, 0) + 1
        label_freq = dict(sorted(label_freq.items()))
        label_freq_array = np.array(list(label_freq.values()))
        label_freq_array = label_freq_array / label_freq_array.sum()
        adjustments = np.log(label_freq_array**tro + 1e-12)
        adjustments = torch.from_numpy(adjustments)
        # adjustments = adjustments.to(self.device)
        print('Logit adjustments: ', adjustments)
        return adjustments
