import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

from options.train_options import args


class ResDown(nn.Module):
    # residual downsample module
    def __init__(self, in_channel=32, out_channel=32,
                 kernel=(3, 3), down_stride=(2, 2), stride=(1, 1), padding=(1, 1), bias=True):
        super(ResDown, self).__init__()
        self.channel = args.feature_num
        self.res = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel,  stride=stride, padding=padding, bias=bias),
            nn.PReLU(),
            nn.Conv2d(self.channel, self.channel, kernel, stride=stride, padding=padding, bias=bias),
            nn.PReLU(),
            nn.Conv2d(self.channel, self.channel, kernel, stride=down_stride, padding=padding, bias=bias),
            nn.Conv2d(self.channel, self.channel, kernel, stride=stride, padding=padding, bias=bias)
        )
        self.down_conv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel, stride=stride, padding=padding, bias=bias),
            nn.Conv2d(self.channel, self.channel, kernel, stride=down_stride, padding=padding, bias=bias),
            nn.PReLU()
        )

    def forward(self, x):
        res = self.res(x)
        x_down = self.down_conv(x)
        out = res + x_down
        return out


class ResUp(nn.Module):
    # residual upsample module
    def __init__(self, in_channel=32, out_channel=32,
                 kernel=(3, 3), up_stride=(2, 2), stride=(1, 1), padding=(1, 1), bias=True):
        super(ResUp, self).__init__()
        self.channel = args.feature_num
        self.res = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel,  stride=stride, padding=padding, bias=bias),
            nn.PReLU(),
            nn.Conv2d(self.channel, self.channel, kernel, stride=stride, padding=padding, bias=bias),
            nn.PReLU(),
            nn.ConvTranspose2d(self.channel, self.channel, kernel, stride=up_stride,
                               padding=padding, output_padding=(1, 1), bias=bias),
            nn.Conv2d(self.channel, out_channel, kernel, stride=stride, padding=padding, bias=bias)
        )
        self.up_conv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel, stride=stride, padding=padding, bias=bias),
            nn.ConvTranspose2d(self.channel, out_channel, kernel, stride=up_stride,
                               padding=padding, output_padding=(1, 1),  bias=bias),
            nn.PReLU()
        )

    def forward(self, x):
        res = self.res(x)
        x_up = self.up_conv(x)
        out = res + x_up
        return out


class ChannelAtt(nn.Module):
    # channel attention module
    def __init__(self, stride, reduction=8, bias=True):
        super(ChannelAtt, self).__init__()
        self.channel = args.feature_num
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale
        self.channel_down = nn.Sequential(
            nn.Conv2d(self.channel, self.channel // reduction, (1, 1), padding=0, bias=bias),
            nn.PReLU()
        )
        # feature upscale --> channel weight
        self.channel_up1 = nn.Sequential(
            nn.Conv2d(self.channel // reduction, self.channel, (1, 1), padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.channel_up2 = nn.Sequential(
            nn.Conv2d(self.channel // reduction, self.channel, (1, 1), padding=0, bias=bias),
            nn.Sigmoid()
        )
        # different resolution to same
        self.up = nn.Sequential(
            nn.ConvTranspose2d(self.channel, self.channel, (3, 3), stride=stride,
                               padding=(1, 1), output_padding=(1, 1), bias=bias),
            nn.PReLU()
        )

    def forward(self, x, y):
        if x.shape[2] < y.shape[2]:
            x = self.up(x)
        fusion = torch.add(x, y)
        fusion = self.channel_down(self.avg_pool(fusion))
        out_x = self.channel_up1(fusion)
        out_y = self.channel_up2(fusion)
        return [out_x, out_y]


class SpatialAtt(nn.Module):
    # spatial attention module
    def __init__(self, stride, kernel=(3, 3), padding=(1, 1), bias=True):
        super(SpatialAtt, self).__init__()
        self.channel = args.feature_num
        self.trans_conv = nn.Sequential(
            # nn.Conv2d(self.channel, self.channel, kernel, stride=(1, 1), padding=padding, bias=bias),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(self.channel, self.channel, kernel_size=(3, 3), stride=(2, 2),
                               padding=padding, output_padding=(1, 1), bias=bias),
            nn.PReLU()
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(self.channel * 2, self.channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=bias),
            # nn.BatchNorm2d(in_channel, eps=1e-5, momentum=0.01, affine=True),
            nn.PReLU()
        )
        self.down = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(2, 2), padding=padding, bias=bias),
            nn.PReLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(self.channel, self.channel, kernel_size=(3, 3), stride=(2, 2),
                               padding=padding, output_padding=(1, 1), bias=bias),
            nn.Sigmoid()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(self.channel, self.channel, kernel_size=(3, 3), stride=(2, 2),
                               padding=padding, output_padding=(1, 1), bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        if x.shape[2] < y.shape[2]:
            x = self.trans_conv(x)
        fusion = torch.cat([x, y], dim=1)
        fusion = self.down(self.conv_fusion(fusion))
        up_x = self.up1(fusion)
        up_y = self.up2(fusion)
        return [up_x, up_y]


class FusionBlock(nn.Module):
    def __init__(self, upscale):
        super(FusionBlock, self).__init__()
        self.channel = args.feature_num
        self.channel_map = ChannelAtt(upscale)
        self.spatial_map = SpatialAtt(self.channel, upscale)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(self.channel, self.channel, (3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1), bias=True),
            nn.ReLU()
        )
        self.one = nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x, y):
        if x.shape[2] < y.shape[2]:
            up_x = self.up(x)
        else:
            up_x = x

        # fusion = torch.cat((up_x, y), dim=1)
        # fusion = self.one(fusion)
        # fusion = up_x + y

        fusion_x = up_x * self.channel_map(x, y)[0] * self.spatial_map(x, y)[0]
        fusion_y = y * self.channel_map(x, y)[1] * self.spatial_map(x, y)[1]
        fusion = fusion_x + fusion_y
        return fusion


class MDA(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, kernel=(3, 3), padding=(1, 1), bias=True):
        super(MDA, self).__init__()
        self.channel = args.feature_num
        self.input = nn.Conv2d(in_channel, self.channel, kernel_size=kernel, stride=(1, 1), padding=padding, bias=bias)
        self.downsample_1x = ResDown(down_stride=(1, 1))
        self.downsample_2x = ResDown(down_stride=(2, 2))
        self.downsample_4x = nn.Sequential(ResDown(down_stride=(2, 2)), ResDown(down_stride=(2, 2)))

        self.upsample_2x = ResUp(up_stride=(2, 2), out_channel=16)
        self.upsample_4x = nn.Sequential(ResUp(up_stride=(2, 2)), ResUp(up_stride=(2, 2), out_channel=8))

        self.fusion_1x = FusionBlock(upscale=1)
        self.fusion_2x = FusionBlock(upscale=2)

        self.output = nn.Sequential(
            nn.Conv2d(104, self.channel, kernel_size=kernel, stride=(1, 1), padding=padding, bias=bias),
            nn.Conv2d(self.channel, out_channel, kernel_size=kernel, stride=(1, 1), padding=padding, bias=bias),
            nn.Tanh()
        )

    def forward(self, ir, vis):
        ir_input = self.input(ir)
        vis_input = self.input(vis)
        ir_1x = self.downsample_1x(ir_input)
        ir_2x = self.downsample_2x(ir_input)
        ir_4x = self.downsample_4x(ir_input)
        vis_1x = self.downsample_1x(vis_input)
        vis_2x = self.downsample_2x(vis_input)
        vis_4x = self.downsample_4x(vis_input)

        fusion_1x_1x = self.fusion_1x(ir_1x, vis_1x)
        fusion_1x_2x = self.fusion_2x(ir_2x, vis_1x)
        fusion_2x_2x = self.fusion_1x(ir_2x, vis_2x)
        fusion_2x_4x = self.fusion_2x(ir_4x, vis_2x)
        fusion_4x_4x = self.fusion_1x(ir_4x, vis_4x)

        fusion_2x_2x = self.upsample_2x(fusion_2x_2x)
        fusion_2x_4x = self.upsample_2x(fusion_2x_4x)
        fusion_4x_4x = self.upsample_4x(fusion_4x_4x)

        fusion = torch.cat([fusion_1x_1x, fusion_1x_2x, fusion_2x_2x, fusion_2x_4x, fusion_4x_4x], dim=1)
        output = self.output(fusion)
        output = output / 2 + 0.5

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
