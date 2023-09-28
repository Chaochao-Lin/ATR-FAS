from functools import partial

import torch
from torch import nn


# 向下压缩卷积
class DownConv(nn.Sequential):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size: int = 3,
                 **kwargs
                 ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
            ),
        )


class DownConvNormAct(nn.Sequential):
    def __init__(self,
                 in_features,
                 out_features,
                 norm: nn.Module = nn.BatchNorm2d,
                 act: nn.Module = nn.ReLU,
                 kernel_size: int = 3,
                 **kwargs
                 ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )


# 实现深度可分离卷积
class DepthWiseConvNorm(nn.Sequential):
    def __init__(self,
                 in_features,
                 kernel_size,
                 norm: nn.Module = nn.BatchNorm2d,
                 **kwargs
                 ):
        super().__init__(
            nn.Conv2d(
                in_features,
                in_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_features
            ),
            norm(in_features),
        )


class DepthWiseConvNormAct(nn.Sequential):
    def __init__(self,
                 in_features,
                 kernel_size,
                 norm: nn.Module = nn.BatchNorm2d,
                 act: nn.Module = nn.ReLU,
                 **kwargs
                 ):
        super().__init__(
            nn.Conv2d(
                in_features,
                in_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_features
            ),
            norm(in_features),
            act(),
        )


DepthWiseConv3X3BnReLU = partial(DepthWiseConvNormAct, kernel_size=3)
DepthWiseConv1X1BnReLU = partial(DepthWiseConvNormAct, kernel_size=1)
DepthWiseConv3X3Bn = partial(DepthWiseConvNorm, kernel_size=3)
DepthWiseConv1X1Bn = partial(DepthWiseConvNorm, kernel_size=1)


class ConvNorm(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features)
        )


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )


# 这是两种保证形状不变的卷积
Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)
Conv1X1Bn = partial(ConvNorm, kernel_size=1)
Conv3X3Bn = partial(ConvNorm, kernel_size=3)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class L2Normalize(nn.Module):
    def __init__(self, dim):
        super(L2Normalize, self).__init__()
        self.dim = dim

    def forward(self, x):
        # return torch.nn.functional.normalize(x, p=2, dim=self.dim)
        norm = torch.norm(x, 2, dim=self.dim, keepdim=True)
        return torch.div(x, norm + 1e-5)


class Mean(nn.Module):
    def __init__(self, dim):
        super(Mean, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class ResidualAdd(nn.Module):
    def __init__(self, block, shortcut=None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x):
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x
