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
from fusion.MDA import MDA, MDAText
from util.image_pool import ImagePool
from options.train_options import args

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

'''CLIP code'''
device = args.devices if torch.cuda.is_available() else "cpu"
CLIP, preprocess = clip.load("RN50", device=device)
torch_resize = Resize([224, 224])


class TextFusion(nn.Module):
    def __init__(self):
        super(TextFusion, self).__init__()
        self.opt = args
        self.text_path = self.opt.text_save

        # load LLaVA
        self.llava = LLavaDescription(self.opt.llava_path, device=self.opt.llava_device, load_8bit=self.opt.load_8bit_llava,
                                      load_4bit=False)

        self.fusion_model = MDAText()

        # load CLIP
        self.ir_clip_pool = ImagePool(self.opt.pool_size)  # 设置为50
        self.vis_clip_pool = ImagePool(self.opt.pool_size)
        self.fusion_clip_pool = ImagePool(self.opt.pool_size)

        # load prompt for llava
        self.prompt_list = {
            1: 'what targets are significant in this image? answer in a sentence',
            2: 'which regions should be noticed in this image? answer in a sentence',
            3: 'which regions have higher contrast in this image? answer in a sentence',
            4: 'briefly describe the images in a sentence.'
        }

    def forward(self, ir, vis, name):

        ir_llava = torch.cat((ir, ir, ir), 1)
        vis_llava = torch.cat((vis, vis, vis), 1)

        for i in range(len(self.prompt_list)):
            qs = list(self.prompt_list.values())[i]
            # read pre-generated description text of ir and vis images.
            ir_captions = self.read_batch_text(name, tag='ir', text_num=list(self.prompt_list.keys())[i])
            vis_captions = self.read_batch_text(name, tag='vis', text_num=list(self.prompt_list.keys())[i])

            text_ir = clip.tokenize(ir_captions).to(device)
            text_vis = clip.tokenize(vis_captions).to(device)

            with torch.no_grad():
                text_ir_features = CLIP.encode_text(text_ir).to(torch.float32)
                text_vis_features = CLIP.encode_text(text_vis).to(torch.float32)

            if i == 0:
                ir_clip = torch.zeros_like(text_ir_features).unsqueeze(1)
                vis_clip = torch.zeros_like(text_vis_features).unsqueeze(1)
                # 在通道维度上拼接
                ir_clip = torch.concatenate((ir_clip, text_ir_features.unsqueeze(1)), dim=1)
                vis_clip = torch.concatenate((vis_clip, text_vis_features.unsqueeze(1)), dim=1)
            else:
                ir_clip = torch.concatenate((ir_clip, text_ir_features.unsqueeze(1)), dim=1)
                vis_clip = torch.concatenate((vis_clip, text_vis_features.unsqueeze(1)), dim=1)

        ir_clip = ir_clip[:, 1:, :]
        vis_clip = vis_clip[:, 1:, :]
        fusion = self.fusion_model(ir, vis, ir_clip, vis_clip)

        fusion_llava = (fusion - fusion.min()) / (fusion.max() - fusion.min())
        fusion_llava = torch.cat((fusion_llava, fusion_llava, fusion_llava), 1)

        ir_CLIP = torch_resize(ir_llava)
        vis_CLIP = torch_resize(vis_llava)
        fusion_CLIP = torch_resize(fusion_llava)

        irtext_irimg, vistext_visimg, fustext_fusimg =0, 0, 0
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

            # similarity between ir text and ir image
            logits_per_image, logits_per_text = CLIP(ir_CLIP, text_ir)
            irtext_irimg_each = logits_per_image.softmax(dim=-1)
            irtext_irimg = irtext_irimg + irtext_irimg_each

            # similarity between vis text and vis image
            logits_per_image, logits_per_text = CLIP(vis_CLIP, text_vis)
            vistext_visimg_each = logits_per_image.softmax(dim=-1)
            vistext_visimg = vistext_visimg + vistext_visimg_each

            # similarity between fusion text and fusion image
            logits_per_image, logits_per_text = CLIP(fusion_CLIP, text_fusion)
            fusiontext_fusionimg_each = logits_per_image.softmax(dim=-1)
            fustext_fusimg = fustext_fusimg + fusiontext_fusionimg_each

        irtext_irimg = irtext_irimg / len(self.prompt_list)
        vistext_visimg = vistext_visimg / len(self.prompt_list)
        fustext_fusimg = fustext_fusimg / len(self.prompt_list)

        probs = {'irtext_irimg': irtext_irimg, 'vistext_visimg': vistext_visimg,
                 'fustext_fusimg': fustext_fusimg}

        return fusion, probs

    def single_prompt_loss(self, img_clip, sentence):
        # calculate the similarity between each sentence and image
        text_ir = clip.tokenize(sentence).to(device)
        # similarity between ir text and fusion image
        logits_per_image, logits_per_text = CLIP(img_clip, text_ir)
        logits = logits_per_image.softmax(dim=-1)

        return logits

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
