from .base_layer import *


class InvertedResidual(nn.Sequential):
    def __init__(self, in_features, out_features, expansion=2):
        expanded_features = in_features * expansion
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        # narrow -> wide 放大channel维度
                        Conv1X1BnReLU(in_features, expanded_features),
                        # wide -> wide  在高纬度上做深度可分离卷积
                        DepthWiseConv3X3BnReLU(expanded_features),
                        # wide -> narrow 还原channel维度
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                    shortcut=Conv1X1BnReLU(in_features, out_features)
                    if in_features != out_features
                    else None,
                ),
                nn.ReLU(),
            )
        )
