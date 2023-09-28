from __future__ import print_function, division
import torch.utils.data
import torch
from .inverted_residual import InvertedResidual
from .base_layer import *


class ResUNet(nn.Module):
    def __init__(self, res_block=InvertedResidual):
        super(ResUNet, self).__init__()
        # 输入适配器[64, 64, 64]->[128, 64, 64]
        self.pre_u_conv = Conv3X3BnReLU(64, 128)
        # 编码器-------------------------------------------------------------
        self.encoder_0 = res_block(128, 64, 1)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_1 = res_block(64, 128, 2)

        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_2 = res_block(128, 256, 2)

        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_3 = res_block(256, 512, 2)

        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_4 = res_block(512, 1024, 2)

        # 解码器---------------------------------------------------------
        self.up_sample_4 = nn.Upsample(scale_factor=2)
        # 不希望解码器有这么高的精度，所以先用适配器降维
        self.adapt_4 = Conv1X1BnReLU(1024, 512)
        self.decoder_4 = res_block(2 * 512, 256, 1)

        self.up_sample_3 = nn.Upsample(scale_factor=2)
        self.decoder_3 = res_block(2 * 256, 128, 1)

        self.up_sample_2 = nn.Upsample(scale_factor=2)
        self.decoder_2 = res_block(2 * 128, 64, 1)

        self.up_sample_1 = nn.Upsample(scale_factor=2)
        self.decoder_1 = res_block(2 * 64, 64, 1)

        # 输出适配器
        self.last_u_conv = Conv3X3BnReLU(64, 256)

    def forward(self, x):
        x = self.pre_u_conv(x)

        e0 = self.encoder_0(x)

        e1 = self.max_pool_1(e0)
        e1 = self.encoder_1(e1)

        e2 = self.max_pool_2(e1)
        e2 = self.encoder_2(e2)

        e3 = self.max_pool_3(e2)
        e3 = self.encoder_3(e3)

        e4 = self.max_pool_4(e3)
        e4 = self.encoder_4(e4)

        d4 = self.up_sample_4(e4)
        d4 = self.adapt_4(d4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.decoder_4(d4)

        d3 = self.up_sample_3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.decoder_3(d3)

        d2 = self.up_sample_2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.decoder_2(d2)

        d1 = self.up_sample_1(d2)
        d1 = torch.cat((e0, d1), dim=1)
        d1 = self.decoder_1(d1)

        return self.last_u_conv(d1)
