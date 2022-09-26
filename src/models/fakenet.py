import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.c3d_resnet18 import C3dResnet18
from models.vgg_transformer import VGGTransformer
from models.vivit import ViViT
from models.c3dr50 import C3DR50
from models.resnet34 import ResNet34
from models.audio_encoder import VGG, VGG2, ResNet, SEResnet
from models.tcn import MultiscaleMultibranchTCN
from models.temporal_transformer import TemporalTransformer


class FakeNet(nn.Module):

    def __init__(self,
                 backbone,
                 img_in_dim,
                 last_dim,
                 frames_per_clip,
                 num_classes=1,
                 fake_classes=1,
                 mode='VA',
                 relu_type='prelu',
                 predict_label=False,
                 predict_fake_label=False,
                 modality_embedding=False,
                 use_losses=[],
                 aud_feat='mfcc',
                 concat_type='concat',
                 normalized='',
                 logit_adjustment=None,
                 weight_decay=1e-4):
        super(FakeNet, self).__init__()

        self.backbone = backbone
        self.img_in_dim = img_in_dim
        self.last_dim = last_dim
        self.frames_per_clip = frames_per_clip
        self.mode = mode
        self.use_losses = use_losses
        self.aud_feat = aud_feat
        self.concat_type = concat_type
        self.normalized = normalized
        self.predict_label = predict_label
        self.predict_fake_label = predict_fake_label
        self.modality_embedding = modality_embedding
        self.num_classes = num_classes
        self.fake_classes = fake_classes
        self.relu_type = relu_type
        self.logit_adjustment = logit_adjustment
        self.weight_decay = weight_decay

        # video
        if 'V' in self.mode:
            self.v_encoder = self._select_backbone('v_encoder',
                                                   self.backbone[0])

        # audio
        if 'A' in self.mode:
            self.a_encoder = self._select_backbone('a_encoder',
                                                   self.backbone[1])

        if self.predict_label:
            if self.mode == 'VA':
                # Normalized
                if self.normalized == 'batchnorm':
                    self.v_normalized = nn.BatchNorm1d(self.last_dim)
                    self.a_normalized = nn.BatchNorm1d(self.last_dim)
                elif self.normalized == 'layernorm':
                    self.v_normalized = nn.LayerNorm(self.last_dim)
                    self.a_normalized = nn.LayerNorm(self.last_dim)

                self.temporal_classifier = self._select_backbone(
                    'temporal_classifier', self.backbone[2])

        # Loss
        self.CELoss = nn.BCEWithLogitsLoss(
        ) if self.num_classes == 1 else nn.CrossEntropyLoss()
        self.FakeTypeLoss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self,
                x,
                alpha=0,
                y=None,
                fake_type_label=None,
                out_feat=False):
        output = {}

        if self.mode == 'V':
            vid, = x
            v_feats = self.forward_vid(vid)
            if out_feat:
                output['vid'] = v_feats
            if self.normalized == '2norm':
                b, t, d = v_feats.shape
                v_feats = rearrange(v_feats, 'b t d -> b (t d)')
                v_feats = F.normalize(v_feats, p=2, dim=-1)
                v_feats = rearrange(v_feats, 'b (t d) -> b t d', d=d)
            logits, cls_feature = self.forward_classification(v_feats)
            output['logits'] = logits
            if out_feat:
                output['cls'] = cls_feature
        elif self.mode == 'VA':
            vid, aud = x
            v_feats = self.forward_vid(vid)
            a_feats = self.forward_aud(aud)

            if out_feat:
                output['vid'] = v_feats
                output['aud'] = a_feats

            if self.predict_label:
                # normalized before concatenate
                if self.normalized:
                    b, fs, d = v_feats.shape
                    v_feats = v_feats.reshape(b * fs, d)
                    a_feats = a_feats.reshape(b * fs, d)

                    if self.normalized == 'batchnorm' or self.normalized == 'layernorm':
                        # batch normalized or layer normalized
                        v_feats = self.v_normalized(v_feats)
                        a_feats = self.a_normalized(a_feats)

                    if self.normalized == '2norm':
                        # 2norm normalized
                        v_feats = F.normalize(v_feats, dim=-1)
                        a_feats = F.normalize(a_feats, dim=-1)

                    v_feats = v_feats.reshape(b, fs, d)
                    a_feats = a_feats.reshape(b, fs, d)

                if self.concat_type == 'minus':
                    feats = v_feats - a_feats
                elif self.concat_type == 'concat':
                    feats = torch.cat((v_feats, a_feats), -1)
                elif self.concat_type == 'cosine':
                    feats = v_feats * a_feats / (
                        torch.norm(v_feats, dim=(1, 2), keepdim=True) *
                        torch.norm(a_feats, dim=(1, 2), keepdim=True))
                if fake_type_label is not None:
                    logits, fake_logits, cls_feature = self.forward_classification(
                        feats, True, alpha)
                    output['logits'] = logits
                    output['fake_logits'] = fake_logits
                else:
                    logits, cls_feature = self.forward_classification(feats)
                    output['logits'] = logits
                if out_feat:
                    output['cls'] = cls_feature
        else:
            raise NotImplementedError

        if y is not None:
            if logits.shape[-1] != y.shape[-1]:
                y = y.squeeze(1)
            else:
                y = y.float()
            if self.logit_adjustment is not None:
                logits = logits + self.logit_adjustment.to(logits.device)
                bce_loss = self.CELoss(logits, y)

                loss_r = 0
                for parameter in self.temporal_classifier.parameters():
                    loss_r += torch.sum(parameter**2)
                bce_loss = bce_loss + self.weight_decay * loss_r
            else:
                bce_loss = self.CELoss(logits, y)

            output['BCE'] = torch.unsqueeze(bce_loss, 0)

        if fake_type_label is not None:
            if fake_logits.shape[-1] != fake_type_label.shape[-1]:
                fake_type_label = fake_type_label.squeeze(1)
            else:
                fake_type_label = fake_type_label.float()
            # fake_idx = [i for i in range(len(y)) if y[i]]
            # print(fake_logits, fake_type_label)
            # print(fake_idx)
            # print(fake_logits[fake_idx], fake_type_label[fake_idx])
            fake_loss = self.FakeTypeLoss(fake_logits, fake_type_label)

            output['FakeType'] = torch.unsqueeze(fake_loss, 0)

        return output

    def forward_vid(self, vid):
        return self.v_encoder(vid)

    def forward_aud(self, aud):
        return self.a_encoder(aud)

    def forward_classification(self, x, predict_fake_label=False, alpha=0):
        return self.temporal_classifier(x, predict_fake_label, alpha)

    def _select_backbone(self, model_type, model_name):
        if model_type == 'v_encoder':
            if model_name == 'c3d_resnet18':
                m = C3dResnet18(in_dim=self.img_in_dim,
                                last_dim=self.last_dim,
                                relu_type=self.relu_type)
            elif model_name == 'vgg_transformer':
                m = VGGTransformer(
                    in_dim=self.img_in_dim,
                    frames_per_clip=self.frames_per_clip,
                    last_dim=self.last_dim,
                    dropout=0.,
                    emb_dropout=0.,
                )
            elif model_name == 'vivit':
                m = ViViT(image_size=224,
                          patch_size=16,
                          num_frames=self.frames_per_clip,
                          in_channels=self.img_in_dim,
                          dim=self.last_dim,
                          depth=3,
                          heads=12)
            elif model_name == 'c3dr50':
                m = C3DR50(in_channels=self.img_in_dim,
                           frames_per_clip=self.frames_per_clip,
                           block_inplanes=[
                               self.last_dim // 16, self.last_dim // 8,
                               self.last_dim // 4, self.last_dim // 2
                           ])
            elif model_name == 'resnet34':
                m = ResNet34(in_channels=self.img_in_dim,
                             num_filters=[
                                 self.last_dim // 8, self.last_dim // 4,
                                 self.last_dim // 2, self.last_dim
                             ])
            else:
                raise NotImplementedError
        elif model_type == 'a_encoder':
            if self.aud_feat == 'mfcc':
                if model_name == 'vgg':
                    m = VGG(
                        last_dim=self.last_dim,
                        temporal_half=True,  # for c3dr50
                    )
                elif model_name == 'seresnet18':
                    m = SEResnet(
                        layers=[2, 2, 2, 2],
                        num_filters=[
                            self.last_dim // 8, self.last_dim // 4,
                            self.last_dim // 2, self.last_dim
                        ],
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
        elif model_type == 'temporal_classifier':

            dim = self.last_dim * 2 if self.concat_type == 'concat' and self.mode == 'VA' else self.last_dim

            if model_name == 'transformer':
                m = TemporalTransformer(
                    frames_per_clip=self.frames_per_clip,
                    num_classes=self.num_classes,
                    predict_fake_label=self.predict_fake_label,
                    modality_embedding=self.modality_embedding,
                    fake_classes=self.fake_classes,
                    dim=dim,
                    depth=1,
                    heads=8,
                    mlp_dim=2048,
                    dropout=0.1,
                    emb_dropout=0.1,
                )
            elif model_name == 'tcn':
                tcn_options = {
                    "num_layers": 4,
                    "kernel_size": [3, 5, 7],
                    "dropout": 0.2,
                    "dwpw": False,
                    "width_mult": 1,
                }
                hidden_dim = 256
                m = MultiscaleMultibranchTCN(
                    input_size=dim,
                    num_channels=[
                        hidden_dim * len(tcn_options["kernel_size"]) *
                        tcn_options["width_mult"]
                    ] * tcn_options["num_layers"],
                    num_classes=self.num_classes,
                    tcn_options=tcn_options,
                    dropout=tcn_options["dropout"],
                    relu_type=self.relu_type,
                    dwpw=tcn_options["dwpw"],
                )
            elif model_name == 'mlp':
                m = nn.Sequential(nn.Flatten(), nn.Linear(dim, dim),
                                  nn.Dropout(0.1), nn.ReLU(),
                                  nn.Linear(dim, dim), nn.Dropout(0.1),
                                  nn.ReLU(), nn.Linear(dim, self.num_classes))
            else:
                raise NotImplementedError

        return m
