import torch
from torch.nn import functional as F


def AVC_loss(v_feats,
             a_feats,
             loss_type='cross',
             temperature=0.07,
             diffculty='min'):
    batch_size, clips, _ = v_feats.shape
    v_feats = F.normalize(v_feats.reshape(batch_size * clips, -1), p=2, dim=1)
    a_feats = F.normalize(a_feats.reshape(batch_size * clips, -1), p=2, dim=1)

    masks = torch.zeros(
        (batch_size * clips, batch_size * clips)).to(v_feats.device)
    for i in range(batch_size):
        masks[i * clips:(i + 1) * clips, i * clips:(i + 1) * clips] = 1

    if loss_type == 'self':
        # V - V
        vv_scores = torch.exp(torch.matmul(v_feats, v_feats.T) / temperature)
        if diffculty == 'min':
            pos_scores = torch.min(
                vv_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
        elif diffculty == 'median':
            pos_scores = torch.median(
                vv_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
        else:
            raise NotImplementedError
        neg_score_sum = torch.sum(vv_scores * (1 - masks), 1)
        v2v_loss = -torch.log(pos_scores / (pos_scores + neg_score_sum))

        # A - A
        aa_scores = torch.exp(torch.matmul(a_feats, a_feats.T) / temperature)
        if diffculty == 'min':
            pos_scores = torch.min(
                aa_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
        elif diffculty == 'median':
            pos_scores = torch.median(
                aa_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
        else:
            raise NotImplementedError
        neg_score_sum = torch.sum(aa_scores * (1 - masks), 1)
        a2a_loss = -torch.log(pos_scores / (pos_scores + neg_score_sum))

        avc_loss = (v2v_loss + a2a_loss).mean()
    elif loss_type == 'cross':
        # V - A
        va_scores = torch.exp(torch.matmul(v_feats, a_feats.T) / temperature)
        if diffculty == 'min':
            pos_scores = torch.min(
                va_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
        elif diffculty == 'median':
            pos_scores = torch.median(
                va_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
        elif diffculty == 'mean':
            pos_scores = torch.mean(
                va_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
        elif diffculty == 'sum':
            pos_scores = torch.sum(
                va_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
        else:
            raise NotImplementedError
        neg_score_sum = torch.sum(va_scores * (1 - masks), 1)
        v2a_loss = -torch.log(pos_scores / (pos_scores + neg_score_sum))

        # A - V
        av_scores = torch.exp(torch.matmul(a_feats, v_feats.T) / temperature)
        if diffculty == 'min':
            pos_scores = torch.min(
                av_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
        elif diffculty == 'median':
            pos_scores = torch.median(
                av_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
        elif diffculty == 'mean':
            pos_scores = torch.mean(
                av_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
        elif diffculty == 'sum':
            pos_scores = torch.sum(
                av_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
        else:
            raise NotImplementedError
        neg_score_sum = torch.sum(av_scores * (1 - masks), 1)
        a2v_loss = -torch.log(pos_scores / (pos_scores + neg_score_sum))

        avc_loss = (v2a_loss + a2v_loss).mean()
    elif loss_type == 'joint':
        # choose hardest positive score to compute loss
        # V - (V, A)
        vv_scores = torch.exp(torch.matmul(v_feats, v_feats.T) / temperature)
        va_scores = torch.exp(torch.matmul(v_feats, a_feats.T) / temperature)
        if diffculty == 'min':
            pos_scores_in_v = torch.min(
                vv_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
            pos_scores_in_a = torch.min(
                va_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
            pos_scores = torch.min(
                torch.stack((pos_scores_in_v, pos_scores_in_a), 1), 1).values
        elif diffculty == 'median':
            pos_scores_in_v = torch.median(
                vv_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
            pos_scores_in_a = torch.median(
                va_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
            pos_scores = torch.median(
                torch.stack((pos_scores_in_v, pos_scores_in_a), 1), 1).values
        elif diffculty == 'mean':
            pos_scores_in_v = torch.mean(
                vv_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
            pos_scores_in_a = torch.mean(
                va_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
            pos_scores = torch.mean(
                torch.stack((pos_scores_in_v, pos_scores_in_a), 1), 1)
        elif diffculty == 'sum':
            pos_scores_in_v = torch.sum(
                vv_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
            pos_scores_in_a = torch.sum(
                va_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
            pos_scores = torch.sum(
                torch.stack((pos_scores_in_v, pos_scores_in_a), 1), 1)
        else:
            raise NotImplementedError
        neg_score_sum = torch.sum(vv_scores * (1 - masks), 1) + torch.sum(
            va_scores * (1 - masks), 1)
        v2va_loss = -torch.log(pos_scores / (pos_scores + neg_score_sum))

        # choose hardest positive score to compute loss
        # A - (A, V)
        aa_scores = torch.exp(torch.matmul(a_feats, a_feats.T) / temperature)
        av_scores = torch.exp(torch.matmul(a_feats, v_feats.T) / temperature)
        if diffculty == 'min':
            pos_scores_in_v = torch.min(
                aa_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
            pos_scores_in_a = torch.min(
                av_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
            pos_scores = torch.min(
                torch.stack((pos_scores_in_a, pos_scores_in_v), 1), 1).values
        elif diffculty == 'median':
            pos_scores_in_v = torch.median(
                aa_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
            pos_scores_in_a = torch.median(
                av_scores[masks.bool()].reshape(batch_size * clips, -1),
                1).values
            pos_scores = torch.median(
                torch.stack((pos_scores_in_a, pos_scores_in_v), 1), 1).values
        elif diffculty == 'mean':
            pos_scores_in_v = torch.mean(
                aa_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
            pos_scores_in_a = torch.mean(
                av_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
            pos_scores = torch.mean(
                torch.stack((pos_scores_in_v, pos_scores_in_a), 1), 1)
        elif diffculty == 'sum':
            pos_scores_in_v = torch.sum(
                aa_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
            pos_scores_in_a = torch.sum(
                av_scores[masks.bool()].reshape(batch_size * clips, -1), 1)
            pos_scores = torch.sum(
                torch.stack((pos_scores_in_v, pos_scores_in_a), 1), 1)
        else:
            raise NotImplementedError
        neg_score_sum = torch.sum(aa_scores * (1 - masks), 1) + torch.sum(
            av_scores * (1 - masks), 1)
        a2av_loss = -torch.log(pos_scores / (pos_scores + neg_score_sum))

        avc_loss = (v2va_loss + a2av_loss).mean()
    return avc_loss


def AVTS_loss(v_feats, a_feats, temperature=0.07):
    batch_size, clips, _ = v_feats.shape
    avts_loss = 0
    for i in range(batch_size):
        v_f = F.normalize(v_feats[i], p=2, dim=1)
        a_f = F.normalize(a_feats[i], p=2, dim=1)

        va_scores = torch.exp(torch.matmul(v_f, a_f.T) / temperature)
        vv_scores = torch.exp(torch.matmul(v_f, v_f.T) / temperature)
        aa_scores = torch.exp(torch.matmul(a_f, a_f.T) / temperature)

        # ignore self dot product
        vv_scores = vv_scores - torch.diag_embed(torch.diag(vv_scores))
        aa_scores = aa_scores - torch.diag_embed(torch.diag(aa_scores))

        # V - (V, A)
        pos_scores = torch.diag(va_scores)
        total_score_sum = torch.sum(va_scores, 1) + torch.sum(vv_scores, 1)
        v_loss = -torch.log(pos_scores / total_score_sum)

        # A - (V, A)
        pos_scores = torch.diag(va_scores)
        total_score_sum = torch.sum(va_scores, 1) + torch.sum(aa_scores, 1)
        a_loss = -torch.log(pos_scores / total_score_sum)

        avts_loss += (v_loss + a_loss).mean() / batch_size

    return avts_loss


# # b, clips, dim
# v_feats = torch.rand((4, 25, 5*512))
# a_feats = torch.rand((4, 25, 5*512))

# # l = AVTS_loss(v_feats, a_feats)
# l = AVC_loss(v_feats, a_feats)
# print(l)
