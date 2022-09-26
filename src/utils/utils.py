## code source - https://github.com/TengdaHan/DPC/blob/master/utils/utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics


def Contrastive(vid_out, aud_out, target, hyper_param):
    batch_size = target.size(0)

    vid_out = torch.flatten(vid_out, -1)
    aud_out = torch.flatten(aud_out, -1)

    dists_square = torch.nan_to_num(
        torch.sum(torch.square(vid_out - aud_out), -1))
    dists = torch.sqrt(dists_square)

    real_loss = (torch.ones_like(target) - target).squeeze(-1) * dists_square
    fake_loss = target.squeeze(-1) * F.relu(
        (torch.ones_like(dists) * hyper_param - dists))**2
    loss = torch.sum(real_loss + fake_loss).mul(1 / batch_size)

    return loss, dists


def Triplet(real_vid_out, fake_vid_out, aud_out, hyper_param):
    batch_size = aud_out.shape[0]
    loss = 0

    for i in range(batch_size):
        real_dist = torch.dist(real_vid_out[i, :].view(-1),
                               aud_out[i, :].view(-1), 2)
        fake_dist = torch.dist(fake_vid_out[i, :].view(-1),
                               aud_out[i, :].view(-1), 2)
        loss += F.relu(real_dist - fake_dist + hyper_param)

    return loss.mul_(1 / batch_size)


def compute_video_level_acc(video_to_logits, video_to_labels):
    # print(video_to_logits.shape, video_to_labels.shape)
    output_batch = torch.stack([
        torch.mean(torch.stack(video_to_logits[video_id]), 0, keepdim=False)
        for video_id in video_to_logits.keys()
    ])
    output_labels = torch.stack(
        [video_to_labels[video_id] for video_id in video_to_logits.keys()])

    sig = nn.Sigmoid()
    acc = (torch.sum((sig(output_batch) > 0.5) == output_labels) /
           output_batch.shape[0]).item()

    return acc


def compute_video_level_auc(video_to_logits, video_to_labels):
    """ "
    Compute video-level area under ROC curve. Averages the logits across the video for non-overlapping clips.

    Parameters
    ----------
    video_to_logits : dict
        Maps video ids to list of logit values
    video_to_labels : dict
        Maps video ids to label
    """
    output_batch = torch.stack([
        torch.mean(torch.stack(video_to_logits[video_id]), 0, keepdim=False)
        for video_id in video_to_logits.keys()
    ])
    output_labels = torch.stack(
        [video_to_labels[video_id] for video_id in video_to_logits.keys()])

    output_batch = output_batch.cpu().numpy()
    output_labels = output_labels.cpu().numpy()

    print(output_batch.shape, output_labels.shape)

    fpr, tpr, _ = metrics.roc_curve(output_labels, output_batch)

    return metrics.auc(fpr, tpr)


def compute_clip_level_acc(video_to_logits, video_to_labels):
    output_batch = torch.cat([
        torch.stack(video_to_logits[video_id]) for video_id in video_to_logits
    ])
    output_labels = torch.cat([
        torch.stack([video_to_labels[video_id]] *
                    len(video_to_logits[video_id]))
        for video_id in video_to_logits
    ])

    sig = nn.Sigmoid()

    acc = (torch.sum((sig(output_batch) > 0.5) == output_labels) /
           output_batch.shape[0]).item()
    return acc


def compute_clip_level_auc(video_to_logits, video_to_labels):
    output_batch = torch.cat([
        torch.stack(video_to_logits[video_id]) for video_id in video_to_logits
    ]).numpy()
    output_labels = torch.cat([
        torch.stack([video_to_labels[video_id]] *
                    len(video_to_logits[video_id]))
        for video_id in video_to_logits
    ]).numpy()

    fpr, tpr, _ = metrics.roc_curve(output_labels, output_batch)
    return metrics.auc(fpr, tpr)


import math


def change_lr_on_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CosineScheduler:

    def __init__(self, lr_ori, epochs):
        self.lr_ori = lr_ori
        self.epochs = epochs

    def adjust_lr(self, optimizer, epoch):
        reduction_ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.lr_ori * reduction_ratio)
