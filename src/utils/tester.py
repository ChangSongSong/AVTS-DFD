import os
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop
from collections import defaultdict

from models.fakenet import FakeNet
from dataset.video_dataset import VideoDataset
from dataset.transforms import ToTensorVideo, NormalizeVideo, NormalizeUtterance
from utils.utils import *


class Tester():
    def __init__(self, config):
        # Parameters
        self.config = config
        self.device = torch.device(
            f"cuda:{config['test']['gpu_ids'][0]}" if torch.cuda.is_available() else 'cpu')
        self.save_dir = config['test']['save_dir']
        self.use_predict = config['test']['predict']
        self.mode = config['test']['mode']

        # Datasets
        transform = {
            'video': Compose(
                [ToTensorVideo(),
                 CenterCrop((config['dataset']['size'], config['dataset']['size'])),
                 NormalizeVideo((0.421,), (0.165,))]
            ),
            'audio': NormalizeUtterance(),
        }
        test_dataset = VideoDataset(
            root=config['dataset']['test'],
            transform=transform,
            mode='test',
            input_mode=self.mode,
            fps=config['dataset']['fps'],
            duration=config['dataset']['duration'],
            aud_feat=config['dataset']['aud_feat'],)
        print(f'test dataset size: {len(test_dataset)}')

        # DataLoader
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config['dataloader']['batch_size'],
            num_workers=config['dataloader']['num_workers'],
            shuffle=False,
        )

        # Model
        self.model = FakeNet(
            backbone=config['model']['backbone'],
            last_dim=config['model']['last_dim'],
            frames_per_clip=int(config['dataset']['fps'] * config['dataset']['duration']),
            img_in_dim=1 if config['dataset']['grayscale'] else 3,
            mode=config['test']['mode'],
            predict_label=config['model']['predict_label'],
            aud_feat=config['dataset']['aud_feat'],
            normalized=config['model']['normalized'],
            fake_classes=len(self.config['dataset']['train_fake_types']),
            predict_fake_label=self.config['model']['predict_fake_label'],
            modality_embedding=self.config['model']['modality_embedding'],
        )

        # resume model
        checkpoint = torch.load(
            self.config['test']['resume'], map_location=self.device)

        if 'module' in list(checkpoint['model'].keys())[0]:
            self.model = torch.nn.DataParallel(self.model, device_ids=config['test']['gpu_ids']).to(self.device)
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
            self.model = torch.nn.DataParallel(self.model, device_ids=config['test']['gpu_ids']).to(self.device)

        # check save dir exists
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def test(self):
        print('\n--- start testing ---')

        video_to_logits = defaultdict(list)
        video_to_probs = defaultdict(list)
        video_to_labels = {}
        video_per_type_to_logits = defaultdict(lambda: defaultdict(list))
        video_per_type_to_labels = defaultdict(dict)

        sig = nn.Sigmoid()

        self.model.eval()
        with torch.no_grad():
            for it, data in enumerate(tqdm(self.test_loader)):
                x, label, vpath = data

                x = list(map(lambda k: k.to(self.device, dtype=torch.float), x))
                label = label.to(self.device)

                if self.use_predict == 'fc':
                    output = self.model(x)
                    logits = output['logits']
                elif self.use_predict == 'dist':
                    output = self.model(x, out_feat=True)
                    v_feats = output['vid']
                    a_feats = output['aud']
                    b, clips, dim = v_feats.shape
                    v_feats = v_feats.reshape(b, -1)
                    a_feats = a_feats.reshape(b, -1)
                    logits = torch.sqrt(torch.nan_to_num(torch.sum(torch.square(v_feats - a_feats), 1)))
                elif self.use_predict == 'cos':
                    v_feats, a_feats = self.model(x, out_feat=True)
                    logits = torch.cosine_similarity(v_feats, a_feats)

                for i in range(len(vpath)):
                    if 'FakeAVCeleb_v1_2' in vpath[i]:
                        video_id = '_'.join(vpath[i].split('/')[-5:])
                        video_type = vpath[i].split('/')[-5]
                    if 'FakeAVCeleb' in vpath[i]:
                        video_id = '_'.join(vpath[i].split('/')[-6:])
                        video_type = vpath[i].split('/')[-6]
                    elif 'FaceForensics' in vpath[i]:
                        video_id = '_'.join(vpath[i].split('/')[-4:])
                        video_type = vpath[i].split('/')[-4]
                    elif 'DeeperForensics' in vpath[i]:
                        video_id = '_'.join(vpath[i].split('/')[-2:])
                        video_type = vpath[i].split('/')[-2]
                    elif 'DFDC' in vpath[i]:
                        video_id = '_'.join(vpath[i].split('/')[-1:])
                        video_type = ''
                    elif 'LRW' in vpath[i]:
                        video_id = '_'.join(vpath[i].split('/')[-1:])
                        video_type = ''
                    else:
                        raise ValueError(f'video type not found in {vpath[i]}')

                    video_id += '_' + str(int(label[i].item()))
                    video_type += '_' + str(int(label[i].item()))

                    video_to_logits[video_id].append(logits[i].view(-1).detach().cpu())
                    video_to_probs[video_id].append(sig(logits[i]).view(-1).detach().cpu())
                    video_to_labels[video_id] = label[i].view(-1).detach().cpu()
                    video_per_type_to_logits[video_type][video_id].append(logits[i].view(-1).detach().cpu())
                    video_per_type_to_labels[video_type][video_id] = label[i].view(-1).detach().cpu()

        auc_video = compute_video_level_auc(video_to_logits, video_to_labels)
        acc_video = compute_video_level_acc(video_to_logits, video_to_labels)
        auc_clip = compute_clip_level_auc(video_to_logits, video_to_labels)
        acc_clip = compute_clip_level_acc(video_to_logits, video_to_labels)


        print(
            f'test clip auc: {auc_clip:.3f}, acc: {acc_clip:.3f}, video auc: {auc_video:.3f}, acc: {acc_video:.3f}')

        precision_video, recall_video, f1score_video = compute_video_level_precision_recall_F1score(video_to_logits, video_to_labels)
        print(f'video precision: {precision_video}')
        print(f'video recall: {recall_video}')
        print(f'video F1 score: {f1score_video}')


        for t in video_per_type_to_logits:
            acc = compute_video_level_acc(video_per_type_to_logits[t], video_per_type_to_labels[t])
            print('dataset: ', t)
            print(f'acc: {acc:.3f}')

        return

    def resume(self):
        checkpoint = torch.load(
            self.config['test']['resume'], map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
