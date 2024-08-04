# -*- coding: utf-8 -*-
import os
import logging
import random
import sys

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler

from tqdm import tqdm
from kornia.losses import MS_SSIMLoss, SSIMLoss

from data.load_data import TrainData
from fusion.model import LlavaFusion, TextFusion
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
        super(Train, self).__init__()
        self.opt = args
        self.llava_device = self.opt.llava_device
        self.device = self.opt.devices
        self.epochs = self.opt.epochs
        self.lr = self.opt.lr

        self.model_save = self.opt.model_save    # folder to save the trained model
        self.model_pth = os.path.join(self.model_save, 'fusion_model.pth')

        self.train_set = TrainData(self.opt)
        self.train_loader = DataLoader(self.train_set, batch_size=self.opt.batch_size, shuffle=True)

        # load fusion model
        self.fusion_model = TextFusion().to(self.device)

        self.loss_update = FusionLoss().to(self.device)
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
            tqdm_loader = tqdm(self.train_loader, file=sys.stdout)

            for batch, (ir, vis, name) in enumerate(tqdm_loader):
                # forward
                ir, vis = ir.to(self.device), vis.to(self.device)

                fusion, probs = self.fusion_model(ir, vis, name)
                image = {'ir': ir, 'vis': vis, 'fusion': fusion}

                # backward
                self.optimizer.zero_grad(set_to_none=True)
                loss_total, loss = self.loss_update(image, probs)

                loss_total.backward()
                self.optimizer.step()

                loss_total_epoch.append(loss_total.item())
                print(f'Current loss at {epoch}th epoch is {loss_total.item():.3f}')

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


class TrainDDP(object):
    def __init__(self):
        super(TrainDDP, self).__init__()
        self.opt = args
        self.llava_device = self.opt.llava_device
        self.device = self.opt.devices
        self.epochs = self.opt.epochs
        self.gpu_num = self.opt.gpu_num
        self.lr = self.opt.lr

        self.model_save = self.opt.model_save    # folder to save the trained model
        self.model_pth = os.path.join(self.model_save, 'fusion_model.pth')

        self.batch_size = self.opt.batch_size
        self.num_workers = 8

        self.loss = []

    def train(self, rank, nprocs):
        self.setup(rank, nprocs)
        self.train_set = TrainData(self.opt)
        self.train_sampler = DistributedSampler(self.train_set, shuffle=True, drop_last=True, seed=seed)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size // (nprocs - 1),
                                       shuffle=False, num_workers=self.num_workers // (nprocs - 1),
                                       drop_last=True, pin_memory=True, sampler=self.train_sampler)

        # load fusion model
        self.fusion_model = DDP(TextFusion().cuda(), device_ids=[self.device], output_device=self.device,
                                find_unused_parameters=False, broadcast_buffers=False)

        self.loss_update = FusionLoss().cuda()
        self.optimizer = Adam(self.fusion_model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        log = logging.getLogger()
        # initialize for ddp
        # self.setup(self.opt.local_rank, self.epochs)
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

        sampler = self.train_loader.sampler
        for epoch in range(self.epochs):
            tqdm_loader = tqdm(self.train_loader, disable=True)

            sampler.set_epoch(epoch)

            self.fusion_model.train()

            for batch, (ir, vis, name) in enumerate(self.train_loader):

                # forward
                ir, vis = ir.cuda(), vis.cuda()

                fusion, probs = self.fusion_model(ir, vis, name)
                image = {'ir': ir, 'vis': vis, 'fusion': fusion}

                # backward
                self.optimizer.zero_grad(set_to_none=True)
                loss_total, loss = self.loss_update(image, probs)

                loss_total.backward()
                self.optimizer.step()

                # loss for DDP
                loss_tensor = torch.tensor([loss_total.item()]).cuda()
                dist.all_reduce(loss_tensor)
                # 计算所有进程的平均损失
                mean_loss = loss_tensor.item() / dist.get_world_size()  # 平均损失

                loss_total_epoch.append(mean_loss)
                self.save_checkpoint(epoch, self.model_save)

                # dist.barrier()
                dist.destroy_process_group()

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

    def setup(self, rank, world_size):
        # 设置主机地址和端口号，这两个环境变量用于配置进程组通信的初始化。
        # MASTER_ADDR指定了负责协调初始化过程的主机地址，在这里设置为'localhost'，
        # 表示在单机多GPU的设置中所有的进程都将连接到本地机器上。
        os.environ['MASTER_ADDR'] = 'localhost'
        # MASTER_PORT指定了主机监听的端口号，用于进程间的通信。这里设置为'12355'。
        # 注意要选择一个未被使用的端口号来进行监听
        os.environ['MASTER_PORT'] = '12355'
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
        # 初始化分布式进程组。
        # 使用NCCL作为通信后端，这是NVIDIA GPUs优化的通信库，适用于高性能GPU之间的通信。
        # rank是当前进程在进程组中的编号，world_size是总进程数（GPU数量），即进程组的大小。
        # dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='env://')
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size, init_method='tcp://127.0.0.1:8005')
        # 为每个进程设置GPU
        # torch.cuda.set_device(rank)

    def cleanup(self):
        dist.destroy_process_group()

    def synchronize_model(self, model, rank, root='temp_model.pth'):
        if rank == 0:
            # 保存模型到文件
            torch.save(model.state_dict(), root)
        torch.distributed.barrier()  # 等待rank=0保存模型

        if rank != 0:
            # 加载模型权重
            model.load_state_dict(torch.load(root))
        torch.distributed.barrier()  # 确保所有进程都加载了模型

        if rank == 0:
            # 删除临时文件
            os.remove(root)



'''
test
'''


def trainDDP(rank, nprocs):
    self.llava_device = self.opt.llava_device
    self.device = self.opt.devices
    self.epochs = self.opt.epochs
    self.gpu_num = self.opt.gpu_num
    self.lr = self.opt.lr

    self.model_save = self.opt.model_save  # folder to save the trained model
    self.model_pth = os.path.join(self.model_save, 'fusion_model.pth')

    self.train_set = TrainData(self.opt)
    self.train_sampler = DistributedSampler(self.train_set, shuffle=True, drop_last=True, seed=seed)
    self.train_loader = DataLoader(self.train_set, batch_size=self.opt.batch_size // (self.opt.num_gpu - 1),
                                   shuffle=False, num_workers=self.opt.num_workers // (self.opt.num_gpu - 1),
                                   drop_last=True, pin_memory=True, sampler=self.train_sampler)

    # load fusion model
    self.fusion_model = DDP(LlavaFusion().cuda(), device_ids=[self.device], output_device=self.device,
                            find_unused_parameters=False, broadcast_buffers=False)

    self.loss_update = FusionLoss().cuda()
    self.optimizer = Adam(self.fusion_model.parameters(), lr=self.lr)
    self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
    self.loss = []


    os.environ['MASTER_ADDR'] = 'localhost'
    # MASTER_PORT指定了主机监听的端口号，用于进程间的通信。这里设置为'12355'。
    # 注意要选择一个未被使用的端口号来进行监听
    os.environ['MASTER_PORT'] = '12355'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # 初始化分布式进程组。
    # 使用NCCL作为通信后端，这是NVIDIA GPUs优化的通信库，适用于高性能GPU之间的通信。
    # rank是当前进程在进程组中的编号，world_size是总进程数（GPU数量），即进程组的大小。
    # dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='env://')
    torch.distributed.init_process_group("gloo", rank=rank, world_size=nprocs, init_method='env://')


    log = logging.getLogger()
    # initialize for ddp
    # self.setup(self.opt.local_rank, self.epochs)
    # 建立文件夹和tensorboard或wandb等
    loss_total_epoch = []

    # TODO: optionally copy weights from a checkpoint
    if os.path.exists(args.model_pth):
        print(f'Loading pre-trained FuseNet checkpoint {args.model_pth}')
        log.info(f'Loading pre-trained checkpoint {args.model_pth}')
        state = torch.load(args.model_pth)
        args.fusion_model.load_state_dict(state['model'])
    else:
        print("No pre-trained model found")

    sampler = args.train_loader.sampler
    for epoch in range(args.epochs):
        tqdm_loader = tqdm(args.train_loader, disable=True)

        sampler.set_epoch(epoch)
        args.fusion_model.train()

        for batch, (ir, vis) in enumerate(tqdm_loader):
            # forward
            ir, vis = ir.cuda(), vis.cuda()

            fusion, probs = args.fusion_model(ir, vis)
            image = {'ir': ir, 'vis': vis, 'fusion': fusion}

            # backward
            args.optimizer.zero_grad(set_to_none=True)
            loss_total, loss = args.loss_update(image, probs)

            loss_total.backward()
            args.optimizer.step()

            # loss for DDP
            loss_tensor = torch.tensor([loss_total.item()]).cuda()
            dist.all_reduce(loss_tensor)
            # 计算所有进程的平均损失
            mean_loss = loss_tensor.item() / dist.get_world_size()  # 平均损失

            loss_total_epoch.append(mean_loss)
            args.save_checkpoint(epoch, args.model_save)

            dist.barrier()

        args.scheduler.step()
        args.loss.append(np.mean(loss_total_epoch))
        print('epoch: %s, loss: %s' % (epoch, np.mean(loss_total_epoch)))





if __name__ == '__main__':
    if args.train_ddp:
        world_size = args.gpu_num
        num_epochs = args.epochs
        mp.spawn(TrainDDP().train, args=(world_size, ), nprocs=world_size, join=True)
        # mp.spawn(trainDDP, args=(world_size,), nprocs=world_size, join=True)

    else:
        Train().train()
