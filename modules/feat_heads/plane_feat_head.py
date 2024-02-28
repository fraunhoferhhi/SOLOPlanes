import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import code

class PlaneFeatHead(nn.Module):
    def __init__(self,
                 in_channels,   #256
                 out_channels,  #64
                 start_level,   #0
                 end_level,     # 0
                 num_classes,   # 64
                 xy_coord = False,
                 ):    #dict(type='GN', num_groups=32, requires_grad=True)),
        super(PlaneFeatHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = num_classes
        self.xy_coord = xy_coord

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = nn.Sequential(
                    nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    bias=False),

                    nn.GroupNorm(num_channels=self.out_channels,
                    num_groups=32),
                    
                    nn.ReLU()
                    )
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.in_channels+2 if i==3 else self.in_channels
                    one_conv = nn.Sequential(
                        nn.Conv2d(
                        chn,
                        self.out_channels,
                        3,
                        padding=1,
                        bias=False),

                        nn.GroupNorm(num_channels=self.out_channels,
                        num_groups=32),

                        nn.ReLU())
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), one_upsample)
                    continue

                one_conv = nn.Sequential(
                    nn.Conv2d(
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    bias=False),

                    nn.GroupNorm(num_channels=self.out_channels,
                    num_groups=32),

                    nn.ReLU())
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.out_channels,
                self.num_classes,
                1,
                padding=0,
                bias = False),

                nn.GroupNorm(num_channels=self.out_channels,
                num_groups=32),

                nn.ReLU(inplace=True))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs):
        assert len(inputs) == (self.end_level - self.start_level + 1)

        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == 3 and self.xy_coord:
                input_feat = input_p
                x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device)
                y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
                x, y = torch.meshgrid(x_range, y_range, indexing='xy')
                y = y.expand([input_feat.shape[0], 1, -1, -1])
                x = x.expand([input_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                input_p = torch.cat([input_p, coord_feat], 1)
                
            feature_add_all_level = feature_add_all_level + self.convs_all_levels[i](input_p)
        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred
