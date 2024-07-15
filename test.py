# -*- coding: utf-8 -*-
import os
import random
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from data.load_data import TestData
from fusion.model import LlavaFusion
from options.test_options import args

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class Test(object):
    def __init__(self, batch_size=1):
        super(Test, self).__init__()
        self.opt = args
        self.device = self.opt.devices

        self.model_save = self.opt.model_save    # folder to save the trained model
        self.model_pth = os.path.join(self.model_save, 'fusion_model.pth')
        self.fusion_save = self.opt.fusion_save

        self.test_set = TestData(self.opt)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size)
        self.color = self.opt.test_color

        # load fusion model
        self.fusion_model = LlavaFusion().to(self.device)
        self.state = torch.load(self.model_pth)
        self.fusion_model.load_state_dict(self.state['model'])

    def test(self):
        self.fusion_model.eval()

        # TODO: optionally copy weights from a checkpoint
        if os.path.exists(self.model_pth):
            print(f'Loading pre-trained FuseNet checkpoint {self.model_pth}')
            # state = torch.load(self.model_pth)
            # self.fusion_model.load_state_dict(state['model'])
        else:
            print("No pre-trained model found")

        # check if the result folder exists
        if os.path.exists(self.fusion_save):
            print(f'fusion results are saved to {self.fusion_save}')
        else:
            os.makedirs(self.fusion_save)
            print(f'fusion results are saved to {self.fusion_save}')

        tqdm_loader = tqdm(self.test_loader, disable=True)

        if self.color:
            for batch, (ir, vis, vis_cb, vis_cr, name) in enumerate(tqdm_loader):
                # forward
                ir, vis = ir.cuda(), vis.cuda()
                print(f'{name[0]}.png')

                fusion, probs = self.fusion_model(ir, vis, name)
                fusion = fusion.cpu().detach().numpy()
                fusion = np.squeeze(fusion)

                fusion_scale = (fusion - fusion.min()) / (fusion.max() - fusion.min())

                # save color image
                color = np.stack((fusion_scale, vis_cb.squeeze(), vis_cr.squeeze()), axis=2)
                color = cv2.cvtColor(np.float32(color), cv2.COLOR_YCrCb2BGR)
                cv2.imwrite(os.path.join(self.fusion_save, f'{name[0]}.png'), color * 255)

        else:
            for batch, (ir, vis, name) in enumerate(tqdm_loader):
                # forward
                ir, vis = ir.cuda(), vis.cuda()

                fusion, probs = self.fusion_model(ir, vis)
                fusion = fusion.cpu().detach().numpy()
                fusion = np.squeeze(fusion)

                fusion_scale = (fusion - fusion.min()) / (fusion.max() - fusion.min())

                # save gray image
                cv2.imwrite(os.path.join(self.fusion_save, f'{name[0]}.png'), fusion_scale * 255)


if __name__ == '__main__':
    Test().test()
