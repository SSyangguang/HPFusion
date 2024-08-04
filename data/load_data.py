import os
import random
from pathlib import Path

import cv2
import kornia
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms

from options.train_options import args as train_args
from options.test_options import args as test_args


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


class TrainData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    def __init__(self, color=True):
        super(TrainData, self).__init__()
        self.opt = train_args
        self.color = color
        self.ir_path = Path(self.opt.train_ir_path)
        self.vis_path = Path(self.opt.train_vis_path)
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(self.ir_path.glob('*')) if x.suffix in IMG_EXTENSIONS]
        self.vis_list = [x for x in sorted(self.vis_path.glob('*')) if x.suffix in IMG_EXTENSIONS]
        # self.ir_list = [x for x in sorted(self.ir_path.glob('*')) if x.suffix in IMG_EXTENSIONS]
        # self.vis_list = [x for x in sorted(self.vis_path.glob('*')) if x.suffix in IMG_EXTENSIONS]

        # self.transform = get_transform(self.opt)

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vis_path = self.vis_list[index]

        assert ir_path.name == vis_path.name, f"Mismatch ir:{ir_path.name} vis:{vis_path.name}."

        file_name = self.return_name(ir_path)

        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        ir = np.array(ir, dtype='float32') / 255.0
        ir = ir[:, :, np.newaxis]

        if self.color:
            vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
            vis = np.array(vis, dtype='float32') / 255.0
            vis = vis[:, :, 0:1]

        else:
            vis = cv2.imread(str(vis_path), cv2.IMREAD_GRAYSCALE)
            vis = np.array(vis, dtype='float32') / 255.0
            vis = vis[:, :, np.newaxis]

        h, w, _ = ir.shape
        # Permute the images to tensor format
        ir = np.transpose(ir, (2, 0, 1))
        vis = np.transpose(vis, (2, 0, 1))
        # ir = kornia.utils.image_to_tensor(ir).type(torch.FloatTensor)
        # vis = kornia.utils.image_to_tensor(vis).type(torch.FloatTensor)
        ir, vis = torch.Tensor(ir), torch.Tensor(vis)
        # crop
        if self.opt.crop:
            ir, vis = self.crop(ir, vis, h, w)
        if self.opt.resize:
            resize = transforms.Resize([self.opt.resize_size, self.opt.resize_size])
            ir = resize(ir)
            vis = resize(vis)

        self.input_ir = ir.clone()
        self.input_vis = vis.clone()

        return self.input_ir, self.input_vis, file_name

    def __len__(self):
        assert len(self.ir_list) == len(self.vis_list)
        return len(self.ir_list)

    def crop(self, train_ir_img, train_vis_img, h, w):
        # Take random crops
        # h, w, _ = train_ir_img.shape
        x = random.randint(0, h - self.opt.crop_size)
        y = random.randint(0, w - self.opt.crop_size)
        train_ir_img = train_ir_img[:, x: x + self.opt.crop_size, y: y + self.opt.crop_size]
        train_vis_img = train_vis_img[:, x: x + self.opt.crop_size, y: y + self.opt.crop_size]

        return train_ir_img, train_vis_img

    @staticmethod
    def return_name(path):
        file_name = str(path).split('.', 1)[0].rsplit("/", 1)[1]
        return file_name


class TestData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    def __init__(self, color=True):
        super(TestData, self).__init__()
        self.opt = test_args
        self.color = color
        self.ir_path = Path(self.opt.test_ir_path)
        self.vis_path = Path(self.opt.test_vis_path)
        # gain infrared and visible images list
        self.ir_list = [x for x in sorted(self.ir_path.glob('*')) if x.suffix in IMG_EXTENSIONS]
        self.vis_list = [x for x in sorted(self.vis_path.glob('*')) if x.suffix in IMG_EXTENSIONS]

        # self.transform = get_transform(self.opt)

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vis_path = self.vis_list[index]

        assert ir_path.name == vis_path.name, f"Mismatch ir:{ir_path.name} vis:{vis_path.name}."

        file_name = self.return_name(ir_path)

        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        ir = np.array(ir, dtype='float32') / 255.0
        ir = ir[:, :, np.newaxis]

        if self.color:
            vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
            vis = np.array(vis, dtype='float32') / 255.0
            vis_Cb = vis[:, :, 1]
            vis_Cr = vis[:, :, 2]
            vis = vis[:, :, 0:1]

        else:
            vis = cv2.imread(str(vis_path), cv2.IMREAD_GRAYSCALE)
            vis = np.array(vis, dtype='float32') / 255.0
            vis = vis[:, :, np.newaxis]

        # Permute the images to tensor format
        ir = np.transpose(ir, (2, 0, 1))
        vis = np.transpose(vis, (2, 0, 1))
        ir, vis = torch.Tensor(ir), torch.Tensor(vis)

        self.input_ir = ir.clone()
        self.input_vis = vis.clone()

        if self.color:
            # Return Cb and Cr channel
            self.vis_Cb = vis_Cb.copy()
            self.vis_Cr = vis_Cr.copy()

            return self.input_ir, self.input_vis, self.vis_Cb, self.vis_Cr, file_name
        else:
            return self.input_ir, self.input_vis, file_name

    def __len__(self):
        assert len(self.ir_list) == len(self.vis_list)
        return len(self.ir_list)

    def crop(self, train_ir_img, train_vis_img):
        # Take random crops
        h, w, _ = train_ir_img.shape
        x = random.randint(0, h - self.opt.patch_size)
        y = random.randint(0, w - self.opt.patch_size)
        train_ir_img = train_ir_img[x: x + self.opt.patch_size, y: y + self.opt.patch_size]
        train_vis_img = train_vis_img[x: x + self.opt.patch_size, y: y + self.opt.patch_size]

        return train_ir_img, train_vis_img

    @ staticmethod
    def return_name(path):
        file_name = str(path).split('.', 1)[0].rsplit("/", 1)[1]
        return file_name


