import torch
from PIL import Image



import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


'''
Test BLIP2 for image description.
'''
'''
from lavis.models import load_model_and_preprocess
# setup device to use
device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"
# load sample image
raw_image = Image.open("/data/yg/data/FMB/test/Infrared/00043.png").convert("RGB")
# display(raw_image.resize((596, 437)))

# Then we load a pre-trained BLIP-2 model with its preprocessors (transforms).
# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

text1 = model.generate({"image": image})
print(text1)
print('-----------------------')

text2 = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)
print(text2)
print('-----------------------')

text3 = model.generate({"image": image, "prompt": "Question: what are the thermal targets in the infrared image? Answer:"})

print(text3)
print('-----------------------')

text4 = model.generate({"image": image, "prompt": "Question: how many people thermal in this infrared image? Answer:"})

print(text4)
print('-----------------------')

text5 = model.generate({"image": image, "prompt": "Question: how many people in this image? Answer:"})

print(text5)
print('-----------------------')
'''

'''
Test LLaVA for image VQA
'''

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "/u2/llm_model/llava-v1.6-mistral-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device_map="cuda:0"
)

# model_path = "liuhaotian/llava-v1.5-7b"
# model_path = "liuhaotian/llava-v1.6-mistral-7b"
# model_path = "liuhaotian/llava-v1.6-vicuna-13b"

# model_path = "/u2/llm_model/llava-v1.6-vicuna-13b"
image_file = "/data/yg/data/FMB/test/Infrared/00043.png"
prompt = "how many people with thermal in this infrared image?"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0.2,
    "top_p": 0.7,
    "num_beams": 1,
    "max_new_tokens": 512,
    "assign": True
})()

eval_model(args)


'''
Test for CLIP
'''
# import numpy as np
# import torch
# import clip
# import cv2
#
# print("Torch version:", torch.__version__)
#
# print(clip.available_models())
#
# model, preprocess = clip.load("ViT-B/32")
# model.cuda().eval()
# # input_resolution = model.visual.input_resolution
# # context_length = model.context_length
# # vocab_size = model.vocab_size
#
# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
#
# import matplotlib
# matplotlib.use('Agg')
# import os
#
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
#
# from collections import OrderedDicth
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# # # test1
# # image = preprocess(Image.open('./test_clip/ir/00055.png')).unsqueeze(0).to(device)
# # text = clip.tokenize(['a people and four cars', 'a people', 'four cars']).to(device)
#
# # test2
# image = preprocess(Image.open('/data/yg/data/FMB/test/Infrared/00043.png')).unsqueeze(0).to(device)
# text = clip.tokenize(['cars and buildings', 'cars', 'a prople', 'budildings and road',
#                       ' a photo of high contrast', ' a photo of low contrast',
#                       'a image of infrared', 'a image of visible']).to(device)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#
#     logits_per_image, logits_per_text = model(image, text)
#     # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#     probs = logits_per_image.cpu().numpy()
#
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]