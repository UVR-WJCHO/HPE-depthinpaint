from __future__ import division

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch

import cv2
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #opt.gpu_ids = 0      # use [] for CPU'

    opt.name = '0_updateNet_5k' # network name

    ### dataset list ###
    # 'inpaint' : obman & self-dataset
    # 'dexterHO' : dexterHO
    opt.dataset_mode = 'dexterHO'

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    for i, data in enumerate(dataset):
        im_RGBD = data['A']
        im_RGB = data['rgb_both']

        with torch.no_grad():
            output = model.forward_test(im_RGBD, im_RGB)

        im_RGBD = np.squeeze(im_RGBD.numpy())
        output = np.squeeze(output.cpu().numpy())


        print("L")


