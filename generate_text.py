# -*- coding: utf-8 -*-
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.load_data import TestData
from llava_model.llava_agent import LLavaDescription
from options.train_options import args


class LlavaDesGen(nn.Module):
    '''
    Pre generate the description for infrared and visible images via LLaVA.
    The name of generated txt files are same with the name of original image files, such as the ir/000.txt and ir/000.png
    '''
    def __init__(self, question_num):
        super(LlavaDesGen, self).__init__()
        self.opt = args
        self.llava_device = self.opt.llava_device
        self.device = 'cuda:0'
        self.epochs = 1
        self.batch_size = 1
        # folder to save the generated description text
        self.prompt_save = self.opt.text_save

        self.test_set = TestData(self.opt)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size)

        # load LLaVA
        self.llava = LLavaDescription(self.opt.llava_path, device=self.opt.llava_device, load_8bit=self.opt.load_8bit_llava,
                                      load_4bit=False)

        # load prompt for llava
        self.prompt = ['what targets are significant in this image? answer in a sentence',
                       'which regions should be noticed in this image? answer in a sentence',
                       'which regions have higher contrast in this image? answer in a sentence',
                       'briefly describe the images in a sentence.'
                       ]
        self.qs_num = question_num

    def forward(self):
        # create folder to save infrared and visible description folder
        ir_folder = os.path.join(self.prompt_save, 'ir', f'prompt{self.qs_num+1}')
        vis_folder = os.path.join(self.prompt_save, 'vis', f'prompt{self.qs_num+1}')
        if not os.path.exists(ir_folder):
            os.makedirs(ir_folder)
            print(f'Pre generated ir description are saved to {ir_folder}')
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)
            print(f'Pre generated vis description are saved to {vis_folder}')

        for epoch in range(self.epochs):
            tqdm_loader = tqdm(self.test_loader, disable=True)

            for batch, (ir, vis, vis_cb, vis_cr, name) in enumerate(tqdm_loader):
                # forward
                ir, vis = ir.cuda(), vis.cuda()
                vis_cb, vis_cr = vis_cb.cuda(), vis_cr.cuda()
                vis_cb = torch.unsqueeze(vis_cb, 0)
                vis_cr = torch.unsqueeze(vis_cr, 0)

                ir_llava = torch.cat((ir, ir, ir), 1)
                vis_llava = torch.cat((vis, vis_cb, vis_cr), 1)

                # 'Describe the target in this infrared image.'
                ir_captions = self.llava.gen_image_caption(ir_llava, qs=self.prompt[self.qs_num])
                vis_captions = self.llava.gen_image_caption(vis_llava, qs=self.prompt[self.qs_num])
                print(f'{name[0]}.png ir text: {ir_captions[0]}')
                print(f'{name[0]}.png vis text: {vis_captions[0]}')

                with open(os.path.join(ir_folder, f'{name[0]}.txt'), 'w') as file:
                    file.write(str(ir_captions[0]))
                with open(os.path.join(vis_folder, f'{name[0]}.txt'), 'w') as file:
                    file.write(str(vis_captions[0]))

        return 0


if __name__ == '__main__':
    LlavaDesGen(1).forward()
    LlavaDesGen(2).forward()
    LlavaDesGen(3).forward()
    LlavaDesGen(4).forward()
