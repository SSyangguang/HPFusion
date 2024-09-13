# -*- coding: utf-8 -*-
import torch

# from .base_model import BaseModel

import random
import clip
from torchvision.transforms import Resize
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from kornia.losses import MS_SSIMLoss, SSIMLoss

from options.train_options import args

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.opt = args

        self.l1_loss = torch.nn.L1Loss().to(self.opt.devices)
        self.mse_loss = nn.MSELoss(reduction='mean').to(self.opt.devices)
        self.ssim_loss = SSIMLoss(window_size=11).to(self.opt.devices)
        self.grad_loss = Gradloss(coeff_grad =20, device=self.opt.devices).to(self.opt.devices)

    def forward(self, image, probs):
        diagonal_matrix = torch.eye(self.opt.batch_size).to(self.opt.devices)

        '''
        # 以batch为基础计算L1 loss
        # loss for clip similarity
        loss_clip_it_fimg = self.l1_loss(probs['irtext_fusimg'], diagonal_matrix)
        loss_clip_vt_fimg = self.l1_loss(probs['vistext_fusimg'], diagonal_matrix)

        loss_clip_ft_iimg = self.l1_loss(probs['fustext_irimg'], diagonal_matrix)
        loss_clip_ft_vimg = self.l1_loss(probs['fustext_visimg'], diagonal_matrix)
        loss_clip = loss_clip_it_fimg + loss_clip_vt_fimg + loss_clip_ft_iimg + loss_clip_ft_vimg
        loss_clip = torch.mean(loss_clip)
        '''

        # 以模态和自己的img text相似度为基础计算L1 loss
        # loss for clip similarity
        loss_clip_ir_fus = self.l1_loss(probs['irtext_irimg'], probs['fustext_fusimg'])
        loss_clip_vis_fus = self.l1_loss(probs['vistext_visimg'], probs['fustext_fusimg'])
        loss_clip = loss_clip_ir_fus + loss_clip_vis_fus
        loss_clip = torch.mean(loss_clip)


        # loss for MSE
        loss_mse_if = self.mse_loss(image['fusion'], image['ir'])
        loss_mse_vf = self.mse_loss(image['fusion'], image['vis'])
        loss_mse = loss_mse_if + loss_mse_vf
        loss_mse = torch.mean(loss_mse)

        # loss for SSIM
        loss_ssim_if = self.ssim_loss(image['fusion'], image['ir'])
        loss_ssim_vf = self.ssim_loss(image['fusion'], image['vis'])
        loss_ssim = loss_ssim_if + loss_ssim_vf
        loss_ssim = torch.mean(loss_ssim)

        # loss for Gradient
        loss_grad, _, _ = self.grad_loss(image['ir'], image['vis'], image['fusion'])

        loss_total = loss_ssim + self.opt.loss_mse * loss_mse \
                     + self.opt.loss_clip * loss_clip + self.opt.loss_grad * loss_grad
        loss = {'mse': loss_mse, 'ssim': loss_ssim, 'clip': loss_clip, 'grad': loss_grad}

        return loss_total, loss


class Gradloss(nn.Module):
    def __init__(self, coeff_int=1, coeff_grad=10, in_max=True, device='cuda'):
        super(Gradloss, self).__init__()
        self.sobelconv = Sobelxy(device=device)
        self.coeff_int = coeff_int
        self.coeff_grad = coeff_grad
        self.in_max = in_max    # intensity max

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]

        if self.in_max:
            x_in_max = torch.max(image_y, image_ir)
        else:
            x_in_max = (image_y + image_ir) / 2.0
        loss_in = F.l1_loss(x_in_max, generate_img)

        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)

        loss_total = self.coeff_int * loss_in + self.coeff_grad * loss_grad
        return loss_total, loss_in, loss_grad


class Sobelxy(nn.Module):
    def __init__(self, device='cuda'):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)
    def forward(self,x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)