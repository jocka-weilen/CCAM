import torch
import torch.nn as nn
import torch.nn.functional as F


def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)
    # tensor([[1.0000, 0.7834],
    #         [0.7834, 1.0000]], grad_fn=<MmBackward0>)
    # torch.clamp 如果 x < min，则 x = min。 如果 x > max，则 x = max。 如果 min ≤ x ≤ max，则 x 保持不变。
    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

# Minimize Similarity, e.g., push representation of foreground and background apart.
class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            # tensor([[4.3158, 1.2225, 1.5015, 1.2456],
            #         [1.3501, 3.4790, 1.3829, 1.3588],
            #         [1.5930, 1.3012, 3.4432, 1.4744],
            #         [1.2589, 1.2417, 1.3422, 4.0942]], grad_fn=<NegBackward0>)
            # tensor(1.9753, grad_fn= < MeanBackward0 >)
            # _ = torch.mean(loss)
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


# Maximize Similarity, e.g., pull representation of background and background together.
class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        # embedded_bg =torch.Size([2, 3072])
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            # tensor([[0.9995, 0.7237, 0.8025, 0.7304],
            #         [0.7237, 0.9995, 0.7338, 0.7192],
            #         [0.8025, 0.7338, 0.9995, 0.7589],
            #         [0.7304, 0.7192, 0.7589, 0.9995]], grad_fn=<ClampBackward1>)
            # 进行降序排序
            _, indices = sim.sort(descending=True, dim=1)
            # rank 表示相似度的排名
            _, rank = indices.sort(dim=1)
            #     rank = rank - 1 调整后，高排名的样本权重更大，模型会更关注它们
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)

if __name__ == '__main__':
    fg_embedding = torch.randn((4, 12))
    bg_embedding = torch.randn((4, 12))
    # print(fg_embedding, bg_embedding)

    # neg_contrast = NegContrastiveLoss(metric='cos')
    # neg_loss = neg_contrast(fg_embedding, bg_embedding)
    # print(neg_loss)

    # pos_contrast = PosContrastiveLoss(metric='cos')
    # pos_loss = pos_contrast(fg_embedding)
    # print(pos_loss)

    examplar = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 4], [4, 2, 1, 3]])

    _, indices = examplar.sort(descending=True, dim=1)
    print(indices)
    _, rank = indices.sort(dim=1)
    print(rank)
    rank_weights = torch.exp(-rank.float() * 0.25)
    print(rank_weights)
