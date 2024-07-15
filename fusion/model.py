# -*- coding: utf-8 -*-
import os
import random
import clip
from torchvision.transforms import Resize
import numpy as np

import torch
import torch.nn as nn

from llava_model.llava_agent import LLavaDescription
from fusion.swinfusion import SwinFusion
from fusion.MDA import MDA
from util.image_pool import ImagePool
from options.train_options import args

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

'''CLIP code'''
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP, preprocess = clip.load("RN50", device=device)
torch_resize = Resize([224, 224])


class LlavaFusion(nn.Module):
    def __init__(self):
        super(LlavaFusion, self).__init__()
        self.opt = args
        self.text_path = self.opt.text_save

        # load LLaVA
        self.llava = LLavaDescription(self.opt.llava_path, device=self.opt.llava_device, load_8bit=self.opt.load_8bit_llava,
                                      load_4bit=False)

        # load fusion model
        self.fusion_model = SwinFusion(img_size=128, window_size=8, img_range=1., depths=[6, 6, 6, 6],
                                       embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=None,
                                       resi_connection='1conv')
        self.fusion_model = MDA()
        # swinfusion中使用了dropout，测试的时候记得加eval

        # load CLIP
        self.ir_clip_pool = ImagePool(self.opt.pool_size)  # 设置为50
        self.vis_clip_pool = ImagePool(self.opt.pool_size)
        self.fusion_clip_pool = ImagePool(self.opt.pool_size)

        # load prompt for llava
        self.prompt = 'describe the image scene.'
        self.prompt_list = {
            2: 'what targets are significant in this image? answer in a sentence',
            4: 'which regions should be noticed in this image? answer in a sentence',
            5: 'which regions have higher contrast in this image? answer in a sentence',
            6: 'briefly describe the images in a sentence.'
        }

    def forward(self, ir, vis, name):
        fusion = self.fusion_model(ir, vis)

        ir_llava = torch.cat((ir, ir, ir), 1)
        vis_llava = torch.cat((vis, vis, vis), 1)
        fusion_llava = (fusion - fusion.min()) / (fusion.max() - fusion.min())
        fusion_llava = torch.cat((fusion_llava, fusion_llava, fusion_llava), 1)

        ir_CLIP = torch_resize(ir_llava)
        vis_CLIP = torch_resize(vis_llava)
        fusion_CLIP = torch_resize(fusion_llava)

        irtext_fusimg, vistext_fusimg = 0, 0
        fustext_visimg, fustext_irimg = 0, 0
        for i in range(len(self.prompt_list)):
            qs = list(self.prompt_list.values())[i]
            # read pre-generated description text of ir and vis images.
            ir_captions = self.read_batch_text(name, tag='ir', text_num=list(self.prompt_list.keys())[i])
            vis_captions = self.read_batch_text(name, tag='vis', text_num=list(self.prompt_list.keys())[i])
            # generate descriptions via the llava
            fusion_captions = self.llava.gen_image_caption(fusion_llava, qs=qs)

            text_ir = clip.tokenize(ir_captions).to(device)
            text_vis = clip.tokenize(vis_captions).to(device)
            text_fusion = clip.tokenize(fusion_captions).to(device)

            # similarity between ir text and fusion image
            logits_per_image, logits_per_text = CLIP(fusion_CLIP, text_ir)
            irtext_fusimg_each = logits_per_image.softmax(dim=-1)
            irtext_fusimg = irtext_fusimg + irtext_fusimg_each

            # similarity between vis text and fusion image
            logits_per_image, logits_per_text = CLIP(fusion_CLIP, text_vis)
            vistext_fusimg_each = logits_per_image.softmax(dim=-1)
            vistext_fusimg = vistext_fusimg + vistext_fusimg_each

            # similarity between ir img and fusion text
            logits_per_image, logits_per_text = CLIP(ir_CLIP, text_fusion)
            fustext_irimg_each = logits_per_image.softmax(dim=-1)
            fustext_irimg = fustext_irimg + fustext_irimg_each

            # similarity between vis img and fusion text
            logits_per_image, logits_per_text = CLIP(vis_CLIP, text_fusion)
            fustext_visimg_each = logits_per_image.softmax(dim=-1)
            fustext_visimg = fustext_visimg + fustext_visimg_each

        irtext_fusimg = irtext_fusimg / len(self.prompt_list)
        vistext_fusimg = vistext_fusimg / len(self.prompt_list)
        fustext_irimg = fustext_irimg / len(self.prompt_list)
        fustext_visimg = fustext_visimg / len(self.prompt_list)

        # read pre-generated description text of ir and vis images.
        # ir_captions = self.read_batch_text(name, tag='ir', text_num=1)
        # vis_captions = self.read_batch_text(name, tag='vis', text_num=1)

        # generate description via the llava
        # ir_captions = self.llava.gen_image_caption(ir_llava)
        # vis_captions = self.llava.gen_image_caption(vis_llava)
        # fusion_captions = self.llava.gen_image_caption(fusion_llava, qs=self.prompt)
        # # fusion_captions2 = self.llava.gen_image_caption(fusion_llava, qs=self.prompt2)
        # print(f'ir text: {ir_captions}')
        # print(f'vis text: {vis_captions}')
        # print(f'vis text: {fusion_captions}')


        '''
        irtext_fusimg = self.all_descri_loss(fusion_CLIP, ir_captions)
        vistext_fusimg = self.all_descri_loss(fusion_CLIP, vis_captions)
        fustext_irimg = self.all_descri_loss(ir_CLIP, fusion_captions)
        fustext_visimg = self.all_descri_loss(vis_CLIP, fusion_captions)
        '''

        '''
        text_ir = clip.tokenize(ir_captions).to(device)
        text_vis = clip.tokenize(vis_captions).to(device)
        text_fusion = clip.tokenize(fusion_captions).to(device)

        # ir_CLIP = torch_resize(self.ir_clip_pool.query(ir_llava))
        # vis_CLIP = torch_resize(self.vis_clip_pool.query(vis_llava))
        # fusion_CLIP = torch_resize(self.fusion_clip_pool.query(fusion_llava))

        # similarity between ir text and fusion image
        logits_per_image, logits_per_text = CLIP(fusion_CLIP, text_ir)
        irtext_fusimg = logits_per_image.softmax(dim=-1)

        # similarity between vis text and fusion image
        logits_per_image, logits_per_text = CLIP(fusion_CLIP, text_vis)
        vistext_fusimg = logits_per_image.softmax(dim=-1)

        # similarity between ir img and fusion text
        logits_per_image, logits_per_text = CLIP(ir_CLIP, text_fusion)
        fustext_irimg = logits_per_image.softmax(dim=-1)

        # similarity between vis img and fusion text
        logits_per_image, logits_per_text = CLIP(vis_CLIP, text_fusion)
        fustext_visimg = logits_per_image.softmax(dim=-1)
        '''

        probs = {'irtext_fusimg': irtext_fusimg, 'vistext_fusimg': vistext_fusimg,
                 'fustext_irimg': fustext_irimg, 'fustext_visimg': fustext_visimg}

        return fusion, probs

    def single_prompt_loss(self, img_clip, sentence):
        # calculate the similarity between each sentence and image
        text_ir = clip.tokenize(sentence).to(device)
        # similarity between ir text and fusion image
        logits_per_image, logits_per_text = CLIP(img_clip, text_ir)
        logits = logits_per_image.softmax(dim=-1)

        return logits

    # def all_descri_loss(self, img_clip, description):
    #     # split each description to sentences separated by the '.'
    #     text_list = description.split('.')
    #     loss_all_des = []
    #     for des in text_list:
    #         loss_single = self.single_prompt_loss(img_clip, des)
    #         loss_all_des = loss_all_des + loss_single
    #
    #     return loss_all_des

    def all_descri_loss(self, img_clip, description):
        loss_all_des = 0
        for i in range(img_clip.shape[0]):
            # split each description to sentences separated by the '.'
            text_list = description[i].split('.')
            loss_all_des_batch = 0
            for des in text_list:
                loss_single = self.single_prompt_loss(img_clip[i, :, :, :], des)
                loss_all_des_batch = loss_all_des_batch + loss_single

            loss_all_des = loss_all_des + loss_all_des_batch
        return loss_all_des

    def read_batch_text(self, name, tag, text_num):
        # read pre-generated description text with batch size
        captions = []
        for i in range(len(name)):
            with open(os.path.join(self.text_path, tag, f'prompt{text_num}', f'{name[0]}.txt'), 'r') as file:
                caption = file.read()
            captions.append(caption)
        return captions

