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


class InpaintDataset(BaseDataset):
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
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory  # HO2H/train
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        #self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        #self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.files_A_rgb = sorted(glob.glob(self.dir_AB + '/rgb' + '/*.*'))
        self.files_obj_rgb = sorted(glob.glob(self.dir_AB + '/rgb_obj' + '/*.*'))
        self.files_depth = sorted(glob.glob(self.dir_AB + '/depth' + '/*.*'))

        datasize = opt.max_dataset_size
        if opt.max_dataset_size is None:
            datasize = int(len(self.files_A_rgb))
        self.files_A_rgb = self.files_A_rgb[:datasize]
        self.files_obj_rgb = self.files_obj_rgb[:datasize]
        self.files_depth = self.files_depth[:datasize]

        transform_list = [ transforms.ToTensor() ]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)

        Datatype:
            A : Hand+Object
            B : Hand

            depth image:
            [:, :, 0] ~ only object
            [:, :, 1] ~ only hand
            [:, :, 2] ~ combined
        """
        # read a image given a random integer index
        idx_hand_only = 1
        idx_combined = 2
        seed = np.random.randint(2147483647)

        idx = index % len(self.files_A_rgb)
        random.seed(seed)
        ori_A_rgb = self.transform(Image.open(self.files_A_rgb[idx]))
        random.seed(seed)
        ori_obj_rgb = self.transform(Image.open(self.files_obj_rgb[idx]))

        random.seed(seed)
        ori_A_depth = self.transform(Image.open(self.files_depth[idx]))

        AB_depth = ori_A_depth.float().numpy()  # (3, 256, 256)
        AB_depth[AB_depth == 0] = 1.0
        AB_depth = (1.0 - AB_depth) * 2.0 - 1.0

        A_depth = AB_depth[idx_combined, :, :]
        B_depth = AB_depth[idx_hand_only, :, :]

        # from hand-only depth, find ROI with (manual threshold) --> GT(body..included) --> manual

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

        # convert to tensor
        A_rgb = transforms.ToTensor()(A_rgb)
        obj_rgb = transforms.ToTensor()(ori_obj_rgb)
        item_A = transforms.ToTensor()(A_rgbd)  # A_depth
        item_B = transforms.ToTensor()(B_depth)

        return {'A': item_A, 'B': item_B, 'A_rgb': A_rgb, 'obj_rgb': obj_rgb, 'A_paths': self.files_A_rgb, 'B_paths': self.files_depth}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.files_A_rgb)
