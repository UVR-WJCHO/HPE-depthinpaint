import torch
from PIL import Image
import numpy as np
import glob
import random
import os
import argparse
import torchvision.transforms as transforms

class ImageDataset():
    def __init__(self, root, datasize=None, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.normalize = transforms.Normalize((0.5,), (0.5,))
        self.unaligned = unaligned

        self.files_A_rgb = sorted(glob.glob(os.path.join(root, '%s/' % mode) + 'rgb' + '/*.*'))
        self.files_B_rgb = sorted(glob.glob(os.path.join(root, '%s/' % mode) + 'rgb_hand' + '/*.*'))
        self.files_depth = sorted(glob.glob(os.path.join(root, '%s/' % mode) + 'depth' + '/*.*'))

        if datasize is None:
            datasize = int(len(self.files_A_rgb))
        self.files_A_rgb = self.files_A_rgb[:datasize]
        self.files_B_rgb = self.files_B_rgb[:datasize]
        self.files_depth = self.files_depth[:datasize]

        npfile = np.zeros((6, 256, 256, datasize))

        for index in range(datasize):
            if index % 1000 == 0:
                print("idx : ", index)
            seed = np.random.randint(2147483647)

            # from rgb data, extract featuremap(fm)
            random.seed(seed)
            ori_A_rgb = self.transform(Image.open(self.files_A_rgb[index % len(self.files_A_rgb)]))
            A_rgb = np.squeeze(ori_A_rgb.float().numpy())

            random.seed(seed)
            ori_A_depth = self.normalize(self.transform(Image.open(self.files_depth[index % len(self.files_A_rgb)])))
            AB_depth = np.squeeze(ori_A_depth.float().numpy())  # (3, 256, 256)

            npfile[:3, :, :, index] = A_rgb
            npfile[3:, :, :, index] = AB_depth

        np.save('RGBD_10k.npy', npfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')  # 200
    parser.add_argument('--decay_epoch', type=int, default=10,
                        help='epoch to start linearly decaying the learning rate to 0')  # 100
    parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/HO2H/', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')

    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')

    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

    parser.add_argument('--load', action='store_true', help='load from checkpoint')
    parser.add_argument('--datasize', type=int, default=10000, help='set the size of dataset(total 141.5k for obman)')

    # About Yolov3
    parser.add_argument("--model_def", type=str, default="YOLOv3/config/yolov3-tiny.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="YOLOv3/weights/yolov3-tiny.weights",
                        help="path to weights file")

    opt = parser.parse_args()

    transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(opt.size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor()]

    ImageDataset(opt.dataroot, datasize=opt.datasize, transforms_=transforms_, unaligned=True)
    print("done")


