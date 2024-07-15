# -*- coding: utf-8 -*-
import torch

# from .base_model import BaseModel

import random
import clip
from torchvision.transforms import Resize
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler

from kornia.losses import MS_SSIMLoss, SSIMLoss

from options.train_options import args

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.opt = args

        self.l1_loss = torch.nn.L1Loss().cuda()
        self.mse_loss = nn.MSELoss(reduction='mean').cuda()
        self.ssim_loss = SSIMLoss(window_size=11).cuda()

    def forward(self, image, probs):
        diagonal_matrix = torch.eye(self.opt.batch_size).cuda()

        # loss for clip similarity
        loss_clip_it_fimg = self.l1_loss(probs['irtext_fusimg'], diagonal_matrix)
        loss_clip_vt_fimg = self.l1_loss(probs['vistext_fusimg'], diagonal_matrix)

        loss_clip_ft_iimg = self.l1_loss(probs['fustext_irimg'], diagonal_matrix)
        loss_clip_ft_vimg = self.l1_loss(probs['fustext_visimg'], diagonal_matrix)
        loss_clip = loss_clip_it_fimg + loss_clip_vt_fimg + loss_clip_ft_iimg + loss_clip_ft_vimg
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

        loss_total = loss_ssim + self.opt.loss_mse * loss_mse + self.opt.loss_clip * loss_clip
        loss = {'mse': loss_mse, 'ssim': loss_ssim, 'clip': loss_clip}

        return loss_total, loss

