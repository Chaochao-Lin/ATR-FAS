from __future__ import print_function, division
import torch.utils.data
import torch
from .inverted_residual import InvertedResidual
from .base_layer import *


class EasyResUNet(nn.Module):
    def __init__(self, res_block=InvertedResidual):
        super(EasyResUNet, self).__init__()
        # 输入适配器[64, 64, 64]->[128, 64, 64]
        self.pre_u_conv = Conv3X3BnReLU(32, 64)
        # 编码器-------------------------------------------------------------
        self.encoder_0 = res_block(64, 32, 1)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_1 = res_block(32, 64, 2)

        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_2 = res_block(64, 128, 2)

        # 解码器---------------------------------------------------------
        self.up_sample_2 = nn.Upsample(scale_factor=2)
        self.adapt_2 = Conv1X1BnReLU(128, 64)
        self.decoder_2 = res_block(2 * 64, 32, 1)

        self.up_sample_1 = nn.Upsample(scale_factor=2)
        self.decoder_1 = res_block(2 * 32, 32, 1)

        # 输出适配器
        self.last_u_conv = Conv3X3BnReLU(32, 256)

    def forward(self, x):
        x = self.pre_u_conv(x)

        e0 = self.encoder_0(x)

        e1 = self.max_pool_1(e0)
        e1 = self.encoder_1(e1)

        e2 = self.max_pool_2(e1)
        e2 = self.encoder_2(e2)

        # ------------------------------------
        d2 = self.up_sample_2(e2)
        d2 = self.adapt_2(d2)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.decoder_2(d2)

        d1 = self.up_sample_1(d2)
        d1 = torch.cat((e0, d1), dim=1)
        d1 = self.decoder_1(d1)

        return self.last_u_conv(d1)
