# -*- coding: utf-8 -*-
import torch

from .base_model import BaseModel

import random
import clip
from torchvision.transforms import Resize
import numpy as np

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

    def forward(self, ir, vis):
        fusion = self.fusion_model(ir, vis)

        ir_llava = torch.cat((ir, ir, ir), 1)
        vis_llava = torch.cat((vis, vis, vis), 1)
        fusion_llava = (fusion - fusion.min()) / (fusion.max() - fusion.min())
        fusion_llava = torch.cat((fusion_llava, fusion_llava, fusion_llava), 1)

        ir_captions = self.llava.gen_image_caption(ir_llava)
        vis_captions = self.llava.gen_image_caption(vis_llava)
        fusion_captions = self.llava.gen_image_caption(fusion_llava)

        text_ir = clip.tokenize(ir_captions).to(device)
        text_vis = clip.tokenize(vis_captions).to(device)
        text_fusion = clip.tokenize(fusion_captions).to(device)

        ir_CLIP = torch_resize(self.ir_clip_pool.query(ir_llava))
        vis_CLIP = torch_resize(self.vis_clip_pool.query(vis_llava))
        fusion_CLIP = torch_resize(self.fusion_clip_pool.query(fusion_llava))

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

        probs = {'irtext_fusimg': irtext_fusimg, 'vistext_fusimg': vistext_fusimg,
                 'fustext_irimg': fustext_irimg, 'fustext_visimg': fustext_visimg}
        caption = {'ir': ir_captions, 'vis': vis_captions, 'fusion': fusion_captions}

        return fusion, probs, caption

