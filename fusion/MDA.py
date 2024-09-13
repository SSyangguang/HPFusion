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


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)

        return attn_output


class Img2Text(nn.Module):
    def __init__(self, in_channel, mid_channel, hidden_dim):
        super(Img2Text, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), x.size().numel() // x.size(0) // self.hidden_dim, self.hidden_dim)
        return x


class CrossBlock(nn.Module):
    def __init__(self, input_dim=256, image2text_dim=32, hidden_dim=256):
        super(CrossBlock, self).__init__()
        self.image2text_dim = image2text_dim
        self.convA_1 = nn.Sequential(
            nn.Conv2d(input_dim*2, args.feature_num, 1),
            nn.PReLU()
        )
        self.convA_2 = nn.Sequential(
            nn.Conv2d(image2text_dim, args.feature_num, 1),
            nn.PReLU()
        )
        self.convB_1 = nn.Sequential(
            nn.Conv2d(input_dim*2, args.feature_num, 1),
            nn.PReLU()
        )
        self.convB_2 = nn.Sequential(
            nn.Conv2d(image2text_dim, args.feature_num, 1),
            nn.PReLU()
        )
        self.cross_attentionA1 = CrossAttention(embed_dim=hidden_dim, num_heads=8)
        self.cross_attentionA2 = CrossAttention(embed_dim=hidden_dim, num_heads=8)
        self.imagef2textfA1 = Img2Text(input_dim, image2text_dim, hidden_dim)
        self.imagef2textfB1 = Img2Text(input_dim, image2text_dim, hidden_dim)
        self.image2text_dim = image2text_dim

    def forward(self, imageA, imageB, text):
        b, c, H, W = imageA.shape

        imageAtotext = self.imagef2textfA1(imageA)
        imageBtotext = self.imagef2textfB1(imageB)

        ca_A = self.cross_attentionA1(text, imageAtotext, imageAtotext)
        imageA_sideout = imageA
        ca_A = torch.nn.functional.adaptive_avg_pool1d(ca_A.permute(0, 2, 1), 1).permute(0, 2, 1)
        ca_A = F.normalize(ca_A, p=1, dim=2)

        ca_A = (imageAtotext * ca_A).view(imageA.shape[0], self.image2text_dim, H, W)
        imageA_sideout = F.interpolate(imageA_sideout, [H, W], mode='nearest')
        ca_A = F.interpolate(ca_A, [H, W], mode='nearest')
        ca_A = self.convA_1(torch.cat(
                (F.interpolate(imageA, [H, W], mode='nearest'), self.convA_2(ca_A) + imageA_sideout), 1))

        ca_B = self.cross_attentionA2(text, imageBtotext, imageBtotext)
        imageB_sideout = imageB
        ca_B = torch.nn.functional.adaptive_avg_pool1d(ca_B.permute(0, 2, 1), 1).permute(0, 2, 1)
        ca_B = F.normalize(ca_B, p=1, dim=2)

        ca_B = (imageBtotext * ca_B).view(imageA.shape[0], self.image2text_dim, H, W)
        imageB_sideout = F.interpolate(imageB_sideout, [H, W], mode='nearest')
        ca_B = F.interpolate(ca_B, [H, W], mode='nearest')
        ca_B = self.convB_1(torch.cat(
                (F.interpolate(imageB, [H, W], mode='nearest'), self.convB_2(ca_B) + imageB_sideout), 1))

        return ca_A, ca_B


class TextPreprocess(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TextPreprocess, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


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

        fusion_x = up_x * self.channel_map(x, y)[0] * self.spatial_map(x, y)[0]
        fusion_y = y * self.channel_map(x, y)[1] * self.spatial_map(x, y)[1]
        fusion = fusion_x + fusion_y
        return fusion


class MDAText(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, kernel=(3, 3), padding=(1, 1), bias=True):
        super(MDAText, self).__init__()
        self.channel = args.feature_num
        self.input = nn.Conv2d(in_channel, self.channel, kernel_size=kernel, stride=(1, 1), padding=padding, bias=bias)
        self.downsample_1x = ResDown(down_stride=(1, 1))
        self.downsample_2x = ResDown(down_stride=(2, 2))
        self.downsample_4x = nn.Sequential(ResDown(down_stride=(2, 2)), ResDown(down_stride=(2, 2)))

        self.upsample_2x = ResUp(up_stride=(2, 2), out_channel=16)
        self.upsample_4x = nn.Sequential(ResUp(up_stride=(2, 2),  out_channel=self.channel), ResUp(up_stride=(2, 2), out_channel=8))

        self.fusion_1x = FusionBlock(upscale=1)
        self.fusion_2x = FusionBlock(upscale=2)

        self.cross_att = CrossBlock(input_dim=self.channel)
        self.q_fusion = nn.Conv1d(4, 1, 1, 1)
        self.text_process = TextPreprocess(1024, out_channel=256)
        self.text_fusion = nn.Conv1d(2, 1, 1, 1)

        self.output = nn.Sequential(
            nn.Conv2d(88, self.channel, kernel_size=kernel, stride=(1, 1), padding=padding, bias=bias),
            nn.Conv2d(self.channel, out_channel, kernel_size=kernel, stride=(1, 1), padding=padding, bias=bias),
            nn.Tanh()
        )

    def forward(self, ir, vis, text_ir, text_vis):
        ir_input = self.input(ir)
        vis_input = self.input(vis)

        text_ir = self.q_fusion(text_ir)
        text_vis = self.q_fusion(text_vis)

        text_ir = self.text_process(text_ir)
        text_vis = self.text_process(text_vis)
        text = self.text_fusion(torch.concatenate((text_ir, text_vis), dim=1))

        # cross attention
        ir_cross, vis_cross = self.cross_att(ir_input, vis_input, text)
        ir_cross, vis_cross = self.cross_att(ir_cross, vis_cross, text)

        ir_1x = self.downsample_1x(ir_cross)
        ir_2x = self.downsample_2x(ir_cross)
        ir_4x = self.downsample_4x(ir_cross)
        vis_1x = self.downsample_1x(vis_cross)
        vis_2x = self.downsample_2x(vis_cross)
        vis_4x = self.downsample_4x(vis_cross)

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
