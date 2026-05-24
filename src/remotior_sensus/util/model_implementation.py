# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2026 Luca Congedo.
# Author: Luca Congedo
# Email: ing.congedoluca@gmail.com
#
# This file is part of Remotior Sensus.
# Remotior Sensus is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# Remotior Sensus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Remotior Sensus. If not, see <https://www.gnu.org/licenses/>.

"""
Model implementation
"""

import collections

try:
    # noinspection PyPackageRequirements
    from sklearn.model_selection import train_test_split
except Exception as error:
    str(error)
    train_test_split = None

from remotior_sensus.core import configurations as cfg

nn_module = None
try:
    import torch
    from torch import nn
    import torch.nn.functional as functional
    import torch.nn.init as init
    nn_module = nn.Module
    from torch.utils.data import DataLoader, TensorDataset
except Exception as error:
    # empty class
    class Module:
        pass


    nn_module = Module
    if cfg.logger is not None:
        cfg.logger.log.error(str(error))

try:
    # noinspection PyPackageRequirements
    from torchvision.models import swin_v2_b, swin_v2_t
    # noinspection PyPackageRequirements
    from torchvision.models.feature_extraction import create_feature_extractor
    # noinspection PyPackageRequirements
    from torchvision.ops import FeaturePyramidNetwork
except Exception as error:
    str(error)


"""
The following code adapts the SwinV2-based semantic segmentation architecture 
from:
- SatlasPretrain models: https://github.com/allenai/satlaspretrain_models
- Torchvision Swin Transformer and FPN: https://github.com/pytorch/vision

License: see original repositories
"""
class SwinBackboneWrapper(nn.Module):
    def __init__(self, swin):
        super().__init__()
        self.swin = swin

    def forward(self, x):
        outputs = []
        for layer in self.swin.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class FPN(nn.Module):
    def __init__(self, backbone_channels):
        super().__init__()
        out_channels = 128
        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=out_channels
        )
        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]

    def forward(self, x):
        inp = collections.OrderedDict(
            [('feat{}'.format(i), el) for i, el in enumerate(x)]
        )
        output = self.fpn(inp)
        return list(output.values())


class Upsample(nn.Module):
    def __init__(self, backbone_channels):
        super().__init__()
        self.in_channels = backbone_channels
        out_channels = backbone_channels[0][1]
        self.out_channels = [(1, out_channels)] + backbone_channels
        layers = []
        depth, ch = backbone_channels[0]
        while depth > 1:
            next_ch = max(ch // 2, out_channels)
            layers.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ))
            ch = next_ch
            depth //= 2
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        output = self.layers(x[0])
        return [output] + x


class PixelSegmentationHead(nn.Module):
    def __init__(self, backbone_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        in_channels = backbone_channels[0][1]
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, feats, targets=None):
        logits = self.layers(feats[0])
        loss = None
        if targets is not None:
            targets = torch.stack(targets, dim=0).long()
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss


class SatlasSegmentationModel(nn.Module):
    def __init__(self, variant, num_classes, band_number):
        super().__init__()

        if variant == cfg.variant_tiny:
            swin = swin_v2_t()
            swin.features[0][0] = nn.Conv2d(
                in_channels=band_number, out_channels=96, kernel_size=(4, 4),
                stride=(4, 4)
            )
            self.out_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
        else:
            swin = swin_v2_b()
            swin.features[0][0] = nn.Conv2d(
                in_channels=band_number, out_channels=128, kernel_size=(4, 4),
                stride=(4, 4)
            )
            self.out_channels = [
                [4, 128],
                [8, 256],
                [16, 512],
                [32, 1024],
            ]
        self.swin = SwinBackboneWrapper(swin)
        self.fpn = FPN(self.out_channels)
        self.upsample = Upsample(self.fpn.out_channels)
        self.head = PixelSegmentationHead(
            backbone_channels=self.upsample.out_channels,
            num_classes=num_classes,
        )

    """
    def forward(self, x, targets=None):
        feats = self.swin(x)
        feats = self.fpn(feats)
        feats = self.upsample(feats)
        outputs, loss = self.head(feats, targets)
        return outputs, loss
    """

    # implement test-time augmentation
    def forward(self, x, targets=None):
        # flip input
        x_hf = torch.flip(x, dims=[-1])
        x_vf = torch.flip(x, dims=[-2])
        combined_input = torch.cat([x, x_hf, x_vf], dim=0)
        # forward pass
        feats = self.swin(combined_input)
        feats = self.fpn(feats)
        feats = self.upsample(feats)
        logits, loss = self.head(feats, targets)
        # separate outputs
        out_orig, out_hf, out_vf = torch.chunk(logits, 3, dim=0)
        # reverse flip output
        out_hf = torch.flip(out_hf, dims=[-1])
        out_vf = torch.flip(out_vf, dims=[-2])
        # average logits
        avg_logits = (out_orig + out_hf + out_vf) / 3.0
        return avg_logits, loss



# ---------------- Utilities ----------------
def make_layer(basic_block, num_basic_block, **kwarg):
    """Stack multiple blocks to form a sequential layer."""
    layers = [basic_block(**kwarg) for _ in range(num_basic_block)]
    return nn.Sequential(*layers)


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize weights for Conv2d, Linear, and BatchNorm layers."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def pixel_unshuffle(x, scale):
    """Pixel unshuffle (downsamples spatially and increases channels)."""
    b, c, hh, hw = x.size()
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale, h, w)

# ---------------- Residual Dense Block ----------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # uses YOUR default_init_weights
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x  # residual scaling


# ---------------- RRDB (Residual-in-Residual) ----------------
class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x  # residual-in-residual scaling


# ---------------- SSrRrDBNet (BasicSR-free) ----------------
class SSrRrDBNet(nn.Module):
    """
    BasicSR-free implementation of RRDBNet used in ESRGAN/SSR models.

    Args:
        num_in_ch (int): input channels (e.g. bands or stacked LR images * 3)
        num_out_ch (int): output channels (e.g. 3 or band_number)
        scale (int): upscaling factor (1, 2, 4, 8, 16)
        num_feat (int): feature channels (default 64)
        num_block (int): number of RRDB blocks (default 23)
        num_grow_ch (int): growth channels inside RDB (default 32)
    """

    def __init__(
        self,
        num_in_ch,
        num_out_ch,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    ):
        super().__init__()
        self.scale = scale

        # Pixel-unshuffle adjustment (matches BasicSR behavior)
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        # First conv
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # Main trunk (23 RRDB blocks)
        self.body = make_layer(
            RRDB,
            num_block,
            num_feat=num_feat,
            num_grow_ch=num_grow_ch
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling layers (ESRGAN-style)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        if self.scale in (8, 16):
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.scale == 16:
                self.conv_up4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Final reconstruction
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # Pixel-unshuffle for scale 1 or 2 (same as original BasicSR)
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        # Shallow features
        feat = self.conv_first(feat)

        # RRDB trunk
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat  # global residual

        # Upsampling (nearest + conv like ESRGAN)
        feat = self.lrelu(
            self.conv_up1(functional.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(functional.interpolate(feat, scale_factor=2, mode="nearest"))
        )

        if self.scale in (8, 16):
            feat = self.lrelu(
                self.conv_up3(functional.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            if self.scale == 16:
                feat = self.lrelu(
                    self.conv_up4(functional.interpolate(feat, scale_factor=2, mode="nearest"))
                )

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out