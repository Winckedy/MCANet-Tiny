"""MCANet-Tiny model definition with selectable attention (SE, CA, None)."""
from typing import List, Optional, Callable
import torch
from torch import nn


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    """Standard Squeeze-and-Excitation block."""
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CoordAtt(nn.Module):
    """Coordinate Attention module."""
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        mip = max(8, inp // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_h * a_w
        return out


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ):
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        layers: List[nn.Module] = [
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      dilation=dilation, groups=groups, bias=False),
            norm_layer(out_planes),
        ]

        if activation_layer is nn.Identity:
            layers.append(activation_layer())
        else:
            layers.append(activation_layer(inplace=True))

        super().__init__(*layers)


class InvertedResidualConfig:
    def __init__(self, input_channels, kernel, expanded_channels, out_channels,
                 attn_type, activation, stride, dilation=1):
        self.input_channels = input_channels
        self.kernel = kernel
        self.expanded_channels = expanded_channels
        self.out_channels = out_channels
        self.attn_type = attn_type          # 'se', 'ca', 'none'
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation


class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(
                cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=activation_layer))
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(
            cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
            stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
            norm_layer=norm_layer, activation_layer=activation_layer))

        # Attention module selection
        if cnf.attn_type == 'se':
            layers.append(SELayer(cnf.expanded_channels, reduction=4))
        elif cnf.attn_type == 'ca':
            layers.append(CoordAtt(cnf.expanded_channels, cnf.expanded_channels))
        # 'none' adds no attention

        layers.append(ConvBNActivation(
            cnf.expanded_channels, cnf.out_channels, kernel_size=1,
            norm_layer=norm_layer, activation_layer=nn.Identity))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MCANetTiny(nn.Module):
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int, num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 dropout_rate: float = 0.2, **kwargs):
        super(MCANetTiny, self).__init__()
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        if not (isinstance(inverted_residual_setting, List) and
                all(isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting)):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels,
                                       kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(last_channel, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mcanet_tiny(pretrained=False, width_mult: float = 0.4, attn_type: str = 'ca', **kwargs) -> MCANetTiny:
    """
    Build MCANet-Tiny with selectable attention type.
    Args:
        pretrained: if True, loads pretrained weights (not available).
        width_mult: width multiplier for channel scaling.
        attn_type: 'ca' (Coordinate Attention), 'se' (Squeeze-and-Excitation), or 'none'.
    """
    if pretrained:
        print("Warning: MCANet-Tiny is a custom architecture. No pretrained weights available. "
              "Using random initialization.")
    inverted_residual_setting = [
        InvertedResidualConfig(16, 3, 16, 16,  attn_type, "RE", 2),
        InvertedResidualConfig(16, 3, 72, 24,  attn_type, "RE", 2),
        InvertedResidualConfig(24, 5, 96, 40,  attn_type, "HS", 2),
        InvertedResidualConfig(40, 5, 120, 48, attn_type, "HS", 1),
        InvertedResidualConfig(48, 5, 288, 96, attn_type, "HS", 2),
        InvertedResidualConfig(96, 5, 576, 96, attn_type, "HS", 1),
    ]
    for cnf in inverted_residual_setting:
        cnf.input_channels = _make_divisible(cnf.input_channels * width_mult, 8)
        cnf.expanded_channels = _make_divisible(cnf.expanded_channels * width_mult, 8)
        cnf.out_channels = _make_divisible(cnf.out_channels * width_mult, 8)
    last_channel_base = 512
    last_channel = _make_divisible(last_channel_base * width_mult, 8)
    model = MCANetTiny(inverted_residual_setting, last_channel, block=InvertedResidual, **kwargs)
    return model