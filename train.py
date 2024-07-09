# -*- coding: utf-8 -*-
import os
import logging
import torch

# from .base_model import BaseModel

import random
import clip
from torchvision.transforms import Resize
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler

from tqdm import tqdm
from kornia.losses import MS_SSIMLoss, SSIMLoss

from data.load_data import TrainData
from fusion.model import LlavaFusion
from loss.loss_fusion import FusionLoss
from options.train_options import args

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class Train(object):
    def __init__(self):
        self.opt = args
        self.llava_device = self.opt.llava_device
        self.device = self.opt.devices
        self.epochs = self.opt.epochs
        self.lr = self.opt.lr

        self.model_save = self.opt.model_save    # folder to save the trained model
        self.model_pth = os.path.join(self.model_save, 'fusion_model.pth')

        self.train_set = TrainData(self.opt)
        if self.opt.train_ddp:
            self.train_sampler = DistributedSampler(self.train_set, shuffle=True, drop_last=True, seed=seed)
            self.train_loader = DataLoader(self.train_set, batch_size=self.opt.batch_size // (self.opt.num_gpu-1),
                                           shuffle=False, num_workers=self.opt.num_workers // (self.opt.num_gpu-1),
                                           drop_last=True, pin_memory=True, sampler=self.train_sampler)
        else:
            self.train_loader = DataLoader(self.train_set, batch_size=self.opt.batch_size)

        # load fusion model
        self.fusion_model = LlavaFusion().cuda()

        self.loss_update = FusionLoss().cuda()
        self.optimizer = Adam(self.fusion_model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.loss = []

    def train(self):
        log = logging.getLogger()
        # 建立文件夹和tensorboard或wandb等
        loss_total_epoch = []

        # TODO: optionally copy weights from a checkpoint
        if os.path.exists(self.model_pth):
            print(f'Loading pre-trained FuseNet checkpoint {self.model_pth}')
            log.info(f'Loading pre-trained checkpoint {self.model_pth}')
            state = torch.load(self.model_pth)
            self.fusion_model.load_state_dict(state['model'])
        else:
            print("No pre-trained model found")

        for epoch in range(self.epochs):
            tqdm_loader = tqdm(self.train_loader, disable=True)

            if self.opt.train_ddp:
                sampler = self.train_loader.sampler

            for batch, (ir, vis) in enumerate(tqdm_loader):

                # forward
                ir, vis = ir.cuda(), vis.cuda()

                fusion, probs, captions = self.fusion_model(ir, vis)
                image = {'ir': ir, 'vis': vis, 'fusion': fusion}

                # backward
                self.optimizer.zero_grad(set_to_none=True)
                loss_total, loss = self.loss_update(image, probs)

                loss_total.backward()
                self.optimizer.step()

                loss_total_epoch.append(loss_total.item())

                self.save_checkpoint(epoch, self.model_save)

            self.scheduler.step()
            self.loss.append(np.mean(loss_total_epoch))
            print('epoch: %s, loss: %s' % (epoch, np.mean(loss_total_epoch)))

    def save_checkpoint(self, epoch, model_folder):
        model_out_path = os.path.join(model_folder, f'fusion_{epoch:04d}_epoch.pth')
        model_final = self.model_pth
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        state = {
            'model': self.fusion_model.state_dict(),
            'train_loss': self.loss,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        torch.save(state, model_final)
        if epoch % 2 == 0:
            torch.save(state, model_out_path)
            print(f'Checkpoint saved to {model_out_path}')




if __name__ == '__main__':
    Train().train()
