# Code for "Infrared and Visible Image Fusion with Hierarchical Human Perception"

Paper is accepted at [ICASSP 2025](https://arxiv.org/abs/2409.09291)


# Infrared and Visible Image Fusion with Hierarchical Human Perception
>Image fusion combines images from multiple domains into one image, containing complementary information from source domains. Existing methods take pixel intensity, texture and high-level vision task information as the standards to determine preservation of information, lacking enhancement for human perception. We introduce an image fusion method, Hierarchical Perception Fusion (HPFusion), which leverages Large Vision-Language Model to incorporate hierarchical human semantic priors, preserving complementary information that satisfies human visual system. We propose multiple questions that humans focus on when viewing an image pair, and answers are generated via the Large Vision-Language Model according to images. The texts of answers are encoded into the fusion network, and the optimization also aims to guide the human semantic distribution of the fused image more similarly to source images, exploring complementary information within the human perception domain. Extensive experiments demonstrate our HPFusoin can achieve high-quality fusion results both for information preservation and human visual enhancement.

<p align="center">
  <img src="img/architecture.png">
</p>


# Fusion Examples

<p align="center">
  <img src="img/results.png">
</p>

# 🔧 Dependencies and Installation

1. Clone this repo and install dependencies.
* torch>=2.1.2
* opencv-python==4.8.0
2.  Download the checkpoint

#### Dependent Models

* (optional)[llava-v1.6-mistral-7b](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b)
  * Only for the training.

* [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)

#### Our Pre-trained Models

Download via the [Google Drive](https://drive.google.com/file/d/1EmpO0zg95VkfMRxwsF8u-8q1sd7wXFvA/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/1EqNPfDSPRx3LRJxiLpXI9Q?pwd=fdn0)

3. Change the pre-trained model path
* LLaVA path: change the '--llava_path' to the folder of your local LLaVA checkpoint.
* CLIP path: The default pre-trained CLIP model for LLaVA-v1.6-mistral-7b is clip-vit-large-patch14-336. If you want to change the pre-trained CLIP model in LLaVA, please set the "mm_vision_tower" to your local CLIP folder in the config.json of LLaVA.
* Our pre-trained model: change the '--model_save' in the train_options.py and test_options.py to your local .pth file.

## ⚡ Quick Inference

```Shell
Usage: 
python test.py [options]
--devices            GPU for fusion network
--llava_device       GPU for the inference of LLaVA
--test_ir_path       Image folder for infrared images
--test_vis_path      Image folder for visible images
--fusion_save        Image folder for saved fusion results
--model_save         Path for the pre-trained model
```

## 🔖 Text Generation
1. (optional) Set the questions in the 37 line of generate_text.py to your own text if you want to ask other questions for LLaVA. If you want to change the number of questions, please change the number of LlavaDesGen(*) in the main().
2. (optional) If you have asked other questions for LLaVA, you should also change the questions in the line of 43 of the model.py.
3. 
```Shell
python generate_text.py [options]
--devices            GPU for fusion network
--llava_device       GPU for the inference of LLaVA
--llava_path         Path for LLaVA checkpoints.
--train_ir_path      Image folder for infrared images
--train_vis_path     Image folder for visible images
--text_save          Path to save the generated texts.
```

## 📧 Contact
If you have any questions, please email `ssyanguang@gmail.com`.
