import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import glob
import random
import numpy as np
import torchvision.transforms as transforms
import cv2
import time


class SelfDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        transform_list = [ transforms.ToTensor() ]
        self.transform = transforms.Compose(transform_list)

        dir_self = "./datasets/self_2020/wOBJ_processed/" + str(opt.phase)
        self.files_self_A_rgbd = sorted(glob.glob(dir_self + '/A_rgbd' + '/*.*'))
        #self.files_self_A_rgb = sorted(glob.glob(dir_self + '/A_rgb' + '/*.*'))

        self.files_self_B_depth = sorted(glob.glob(dir_self + '/B_d' + '/*.*'))
        self.files_self_obj_rgb = sorted(glob.glob(dir_self + '/obj_rgb' + '/*.*'))

        datasize = opt.max_dataset_size
        if opt.max_dataset_size is None:
            datasize = int(len(self.files_self_A_rgbd))

        self.files_self_A_rgbd = self.files_self_A_rgbd[:datasize]
        #self.files_self_A_rgb = self.files_self_A_rgb[:datasize]
        #self.files_self_A_depth = self.files_self_A_depth[:datasize]
        self.files_self_B_depth = self.files_self_B_depth[:datasize]
        self.files_self_obj_rgb = self.files_self_obj_rgb[:datasize]


    def __getitem__(self, index):
        # self dataset, rgb is already masked by depth
        idx = index % len(self.files_self_A_rgbd)

        A_rgbd = np.array(Image.open(self.files_self_A_rgbd[idx]))
        B_d = np.array(Image.open(self.files_self_B_depth[idx]))

        A_rgb = A_rgbd[:, :, :-1] / 255.0
        A_d = A_rgbd[:, :, -1] / 255.0
        B_d = B_d / 255.0

        A_d = A_d * 2.0 - 1.0
        B_d = B_d * 2.0 - 1.0

        A_rgbd = np.concatenate((A_rgb, np.expand_dims(A_d, axis=-1)), axis=-1)

        A_rgbd = self.transform(A_rgbd)
        B_d = self.transform(B_d)
        obj_rgb = self.transform(Image.open(self.files_self_obj_rgb[idx]))
        A_path = self.files_self_A_rgbd

        return {'A': A_rgbd, 'B': B_d, 'rgb_obj': obj_rgb, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.files_self_A_rgbd)

