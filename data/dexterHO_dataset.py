import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import glob
import random
import numpy as np
import torchvision.transforms as transforms
import cv2


def resize_and_crop_np(img, crop_h, crop_w, top, left):
    # for numpy array from tensor, C * W * H
    img = img[:, int(top):int(top) + crop_h, int(left):int(left) + crop_w]
    img = img.transpose((1, 2, 0))
    return cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)


# only for testing
class DexterHOdataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.dir_AB = 'D:/Research_2020/HPF_handtracker/dataset/DexterHO/Occlusion'

        self.files_A_rgb = sorted(glob.glob(self.dir_AB + '/c2d' + '/*.*'))
        self.files_depth = sorted(glob.glob(self.dir_AB + '/depth' + '/*.*'))

        datasize = opt.max_dataset_size
        if opt.max_dataset_size is None:
            datasize = int(len(self.files_A_rgb))
        self.files_A_rgb = self.files_A_rgb[:datasize]
        self.files_depth = self.files_depth[:datasize]

        transform_list = [ transforms.ToTensor() ]
        self.transform = transforms.Compose(transform_list)


    def __getitem__(self, index):
        # [target]
        # rgb : 0 ~ 1
        # depth : closer ~ smaller, 0 ~ 1
        # to 256, 256

        idx = index % len(self.files_A_rgb)

        A_rgb = cv2.imread(self.files_A_rgb[idx])   # 240, 320, 3
        A_d = cv2.imread(self.files_depth[idx], cv2.IMREAD_UNCHANGED)


        ## check value range
        # rgb : 0 ~ 1
        # depth :
        #   background : 32001
        #   closer = smaller, 1600 ~ 270


        A_d[A_d == 32001] = 0.0
        A_d[A_d > 600] = 0.0
        A_d = A_d / np.max(A_d)

        A_rgb = A_rgb[40:-40, 80:-80, :]
        A_d = A_d[40:-40, 80:-80]

        A_rgb = A_rgb / 255.0

        A_rgb = cv2.resize(A_rgb, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        A_d = cv2.resize(A_d, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

        ## find ROI and crop
        """
               B_depth_tmp = B_depth
               margin = 50
               B_depth_tmp[:margin, :] = 0
               B_depth_tmp[-margin:, :] = 0
               B_depth_tmp[:, :margin] = 0
               B_depth_tmp[:, -margin:] = 0
               nonzero_idx = np.nonzero(B_depth_tmp)
               row_mean = np.mean(nonzero_idx[0])
               col_mean = np.mean(nonzero_idx[1])

               height, width = 160, 160
               top, left = max(0, row_mean - height / 2), max(0, col_mean - width / 2)
               top, left = min(256, top), min(256, left)

               # crop given data
               A_rgb = ori_A_rgb.float().numpy()
               ori_obj_rgb = ori_obj_rgb.float().numpy()

               A_rgb = resize_and_crop_np(A_rgb, height, width, top, left)
               ori_obj_rgb = resize_and_crop_np(ori_obj_rgb, height, width, top, left)

               A_depth = A_depth[int(top):int(top) + height, int(left):int(left) + width]
               A_depth = cv2.resize(A_depth, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

               A_rgbd = np.concatenate((A_rgb, np.expand_dims(A_depth, axis=-1)), axis=-1)

               B_depth = B_depth[int(top):int(top) + height, int(left):int(left) + width]
               B_depth = cv2.resize(B_depth, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
               """

        ## re-order channels and create input
        A_rgbd = np.concatenate((A_rgb, np.expand_dims(A_d, axis=-1)), axis=-1)

        # convert to tensor
        item_A = transforms.ToTensor()(A_rgbd)  # A_depth

        A_path = self.files_A_rgb

        return {'A': item_A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.files_A_rgb)
