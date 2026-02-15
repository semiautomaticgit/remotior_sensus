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
