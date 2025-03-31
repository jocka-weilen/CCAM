from .resnet import *
import torch.nn as nn
import torch
import torch.nn.functional as F
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetSeries(nn.Module):
    def __init__(self, pretrained):
        super(ResNetSeries, self).__init__()

        if pretrained == 'supervised':
            print(f'Loading supervised pretrained parameters!')
            model = resnet50(pretrained=True)
        #     使用 mocov2
        elif pretrained == 'mocov2':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = torch.load('moco_r50_v2-e3b0c442.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif pretrained == 'detco':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = torch.load('detco_200ep.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        # print("recent model ")
        # print(model)
        # quit()
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        x2 = self.layer4(x1)
        return torch.cat([x2, x1], dim=1)

class Disentangler(nn.Module):
    def __init__(self, cin):
        super(Disentangler, self).__init__()
        # fai(.)
        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)

    def forward(self, x, inference=False):
        # x.shape = torch.Size([2, 3072, 28, 28])
        N, C, H, W = x.size()
        if inference:
            ccam = self.bn_head(self.activation_head(x))
        else:
            # 生成 cam
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))

        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        # 矩阵相乘 得到前景和背景
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]
        # 返回 [N,C]    [N,C]     [N, 1, H * W]
        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam


class Network(nn.Module):
    # cin = 2048 + 1024
    def __init__(self, pretrained='mocov2', cin=None):
        super(Network, self).__init__()
        #         x1 = self.layer3(x)
        #         x2 = self.layer4(x1)
        #         return torch.cat([x2, x1], dim=1)
        self.backbone = ResNetSeries(pretrained=pretrained)

        self.ac_head = Disentangler(cin)
        self.from_scratch_layers = [self.ac_head]

    def forward(self, x, inference=False):
        # feats= 第三层和第四层 进行 cat
        feats = self.backbone(x)
        #  [N,C]    [N,C]     [N, 1, H * W]
        fg_feats, bg_feats, ccam = self.ac_head(feats, inference=inference)

        return fg_feats, bg_feats, ccam

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups


def get_model(pretrained, cin=2048+1024):
    return Network(pretrained=pretrained, cin=cin)
