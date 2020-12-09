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


class Self2DDataset(BaseDataset):
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

        # dir_self = "./datasets/self_2020/processed/" + str(opt.phase)
        # self.files_self_A_rgbd = sorted(glob.glob(dir_self + '/A_rgbd' + '/*.*'))
        # self.files_self_A_rgb = sorted(glob.glob(dir_self + '/A_rgb' + '/*.*'))
        #
        # #self.files_self_A_depth = sorted(glob.glob(dir_self + '/A_d' + '/*.*'))
        # self.files_self_B_depth = sorted(glob.glob(dir_self + '/B_d' + '/*.*'))
        # self.files_self_obj_rgb = sorted(glob.glob(dir_self + '/obj_rgb' + '/*.*'))

        datasize = opt.max_dataset_size
        if opt.max_dataset_size is None:
            datasize = int(len(self.files_self_A_rgbd))

        # self.files_self_A_rgbd = self.files_self_A_rgbd[:datasize]
        # #self.files_self_A_rgb = self.files_self_A_rgb[:datasize]
        # #self.files_self_A_depth = self.files_self_A_depth[:datasize]
        # self.files_self_B_depth = self.files_self_B_depth[:datasize]
        # self.files_self_obj_rgb = self.files_self_obj_rgb[:datasize]

        self.A_pair = np.load('D:/Research_2020/tensorflow-pix2pix-light/pair_A_RGBD_430.npy', mmap_mode='r')
        self.B_pair = np.load('D:/Research_2020/tensorflow-pix2pix-light/pair_B_DM_430.npy', mmap_mode='r')
        self.A_pair = self.A_pair[:datasize, :, :, :]
        self.B_pair = self.B_pair[:datasize, :, :, :]

        self.len_pairs = self.A_pair.shape[0]
        # A_pair_1 = np.load('D:/Research_2020/tensorflow-pix2pix-light/pair_A_RGBD_430_1.npy', mmap_mode='r')
        # B_pair_1 = np.load('D:/Research_2020/tensorflow-pix2pix-light/pair_B_DM_430_1.npy', mmap_mode='r')

        self.p_total = np.random.permutation(self.len_pairs)


    def __getitem__(self, index):
        # self dataset, rgb is already masked by depth
        idx = index % self.len_pairs

        A_rgbd_np = self.A_pair[self.p_total[idx]]
        B_np = self.B_pair[self.p_total[idx]]

        rgb = A_rgbd_np[:, :, :-1]
        mask = B_np[:, :, 1]
        rgb[mask==0, 0] = 0
        rgb[mask == 0, 1] = 0
        rgb[mask == 0, 2] = 0

        A_rgbd = self.transform(A_rgbd_np)
        B_d = self.transform(B_np)

        obj_rgb = self.transform(rgb)

        #A_rgb = self.transform(Image.open(self.files_self_A_rgb[idx]))
        #A_rgb = A_rgbd[:-1, :, :]
        #A_d = self.transform(Image.open(self.files_self_A_depth[idx]))

        A_path = None

        return {'A': A_rgbd, 'B': B_d, 'rgb_obj': obj_rgb, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.len_pairs

