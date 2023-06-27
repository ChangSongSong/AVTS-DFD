import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
from collections import OrderedDict

from models.losses.ntxent import NTXentLoss
from models.losses.nce import AVC_loss, AVTS_loss
from models.c3d_resnet18 import C3dResnet18
from models.vgg_transformer import VGGTransformer
from models.vivit import ViViT
from models.resnet18 import ResNet18
from models.c3dr50 import C3DR50
from models.audio_encoder import SEResnet, VGG, VGG2, ResNet
from models.temporal_transformer import TemporalTransformerWithoutMLP


class AVTSNet(nn.Module):
    def __init__(self, backbone, last_dim, frames_per_clip, relu_type = 'prelu', predict_label=False, use_losses=[], aud_feat='mfcc', setting='', img_in_dim=1):
        super(AVTSNet, self).__init__()

        self.backbone = backbone
        self.img_in_dim = img_in_dim
        self.last_dim = last_dim
        self.relu_type = relu_type
        self.frames_per_clip = frames_per_clip
        self.use_losses=use_losses
        self.aud_feat = aud_feat
        self.setting = setting

        # video
        self.v_encoder = self._select_backbone('v_encoder', self.backbone[0])

        # audio
        self.a_encoder = self._select_backbone('a_encoder', self.backbone[1])

        # Loss
        if 'NTXent' in self.use_losses:
            self.NTXentLoss = NTXentLoss(temperature=0.07)
        if 'L2' in self.use_losses:
            self.l2loss = nn.MSELoss()

    def forward(self, vid, aud, alpha=0, y=None, out_feat=False):
        output = {}

        # for self-supervised
        batch_size, clips = vid.shape[:2]
        vid = rearrange(vid, 'b clips ... -> (b clips) ...')
        aud = rearrange(aud, 'b clips ... -> (b clips) ...')

        v_feats, a_feats = self.forward_feat(vid, aud)

        if out_feat:
            output['vid'] = v_feats
            output['aud'] = a_feats

        if y is not None:
            if 'NTXent' in self.use_losses:
                if self.setting == 'AVTS':
                    v_feats = v_feats.reshape(batch_size, clips, -1)
                    a_feats = a_feats.reshape(batch_size, clips, -1)

                    feats = torch.cat((v_feats, a_feats), 1)
                    labels = torch.FloatTensor(torch.arange(clips).repeat(2).float()).to(feats.get_device())
                    metric_loss = 0
                    # only compute loss in the same video
                    for i in range(batch_size):
                        metric_loss += self.NTXentLoss(feats[i], labels) / batch_size
                    output['NTXent_AVTS'] = torch.unsqueeze(metric_loss, 0)
                else:
                    raise NotImplementedError

        return output

    def forward_feat(self, vid, aud):
        return self.forward_vid(vid), self.forward_aud(aud)

    def forward_vid(self, vid):
        return self.v_encoder(vid)

    def forward_aud(self, aud):
        return self.a_encoder(aud)

    def _select_backbone(self, model_type, model_name):
        if model_type == 'v_encoder':
                if model_name == 'c3d_resnet18':
                    m = C3dResnet18(in_dim=self.img_in_dim, last_dim=self.last_dim, relu_type=self.relu_type)
                elif model_name == 'vgg_transformer':
                    m = VGGTransformer(
                                        in_dim=self.img_in_dim,
                                        frames_per_clip=self.frames_per_clip,
                                        last_dim=self.last_dim,
                                        dropout = 0.,
                                        emb_dropout = 0.,)
                elif model_name == 'vivit':
                    m =  ViViT(
                                image_size=224,
                                patch_size=16,
                                num_frames=self.frames_per_clip,
                                in_channels=self.img_in_dim,
                                dim=self.last_dim,
                                depth=3,
                                heads=12)
                elif model_name == 'resnet18':
                    m = ResNet18(
                                in_channels=self.img_in_dim,
                                num_filters=[self.last_dim//8, self.last_dim//4, self.last_dim//2, self.last_dim])
                elif model_name == 'c3dr50':
                    m = C3DR50(
                        block_inplanes=[self.last_dim//16, self.last_dim//8, self.last_dim//4, self.last_dim//2],
                        frames_per_clip=self.frames_per_clip,
                        in_channels=self.img_in_dim
                    )
                else:
                    raise NotImplementedError
        elif model_type == 'a_encoder':
            if self.aud_feat == 'mfcc':
                if model_name == 'vgg':
                    m = VGG(
                            last_dim=self.last_dim,
                            # last_avg=(self.setting=='AVC'),
                            temporal_half=True, # for c3dr50
                            )
                elif model_name == 'seresnet18':
                    m = SEResnet(
                                layers=[2, 2, 2, 2],
                                num_filters=[self.last_dim//8, self.last_dim//4, self.last_dim//2, self.last_dim],
                                avg_time=(self.setting=='AVC')
                            )
                else:
                    raise NotImplementedError
            elif self.aud_feat == 'melspectrogram':
                if model_name == 'vgg':
                    m = VGG2(last_dim=self.last_dim)
                elif model_name == 'resnet18':
                    m = ResNet(frames_per_clip=self.frames_per_clip)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError(f'{self.aud_feat}')
        return m
