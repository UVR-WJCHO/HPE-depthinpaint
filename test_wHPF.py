from __future__ import division

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch

import cv2
import numpy as np
#from HPF_module import HPF_module


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.gpu_ids = [0]      # use [] for CPU'
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    opt.name = 'trial_6_yolo' # network name

    ### dataset list ###
    # 'inpaint' : obman & self-dataset
    # 'dexterHO' : dexterHO
    opt.dataset_mode = 'dexterHO'

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    for i, data in enumerate(dataset):
        im_RGBD = data['A'].to(device)
        im_RGBD = im_RGBD.type(torch.cuda.FloatTensor)

        with torch.no_grad():
            output = model.forward_test(im_RGBD)

        im_RGBD = np.squeeze(im_RGBD.cpu().numpy())
        output = np.squeeze(output.cpu().numpy())

        im_RGBD = im_RGBD.transpose((1, 2, 0))

        im_RGB = np.uint8(im_RGBD[:, :, :-1] * 255)
        im_D = im_RGBD[:, :, -1]
        cv2.imshow("input rgb", im_RGB)
        cv2.imshow("input depth", im_D)
        cv2.imshow("output d", output)
        cv2.waitKey(1)

        output = np.uint8((output + 1) * 255 / 2)
        name_rgb = "results/" + opt.name + "/test_DexterHO/rgb/" + str(i) + ".png"
        name_output = "results/" + opt.name + "/test_DexterHO/output/" + str(i) + ".png"
        cv2.imwrite(name_rgb, im_RGB)
        cv2.imwrite(name_output, output)



