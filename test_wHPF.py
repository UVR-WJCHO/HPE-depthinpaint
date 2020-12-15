from __future__ import division

import os
from options.test_options import TestOptions
from data import create_dataset
from inpaint_models import create_model
import torch
import time
import torchvision.transforms as transforms

import cv2
import numpy as np
from HPF_module import HPF_module


if __name__ == '__main__':
    flag_net = True

    # define hand tracker first(occur error if inpaint module first)
    hpf = HPF_module()

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.gpu_ids = [0]      # use [] for CPU'
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    opt.name = 'trial_7_update' # network name

    ### dataset list ###
    # 'inpaint' : obman & self-dataset
    # 'dexterHO' : dexterHO
    opt.dataset_mode = 'dexterHO'

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # DexterHO occlusion
    crop_h = 25
    crop_w = 65

    while (1):
        key = cv2.waitKey(1)

        # dexterHO (240, 320)
        input = hpf._get_input()
        if input is None:
            break
        dimg, c2dimg = input

        #dimg = np.flip(dimg, 1)
        #c2dimg = np.flip(c2dimg, 1)

        dimg[dimg > 500] = 0
        dimg_inp = np.copy(dimg)

        dimg_norm = 255 * dimg / np.max(dimg)
        dimg_norm_c = np.copy(dimg_norm)

        d_max = np.max(dimg)
        d_min = np.min(dimg[np.nonzero(dimg)])

        if flag_net:
            # normalize range : rgb 0~1, depth -1~1
            c2dimg_norm = np.copy(c2dimg)
            c2dimg_norm = c2dimg_norm / 255.0
            dimg_norm = (dimg_norm / 255.0) * 2.0 - 1.0

            # crop & resize input image
            c2dimg_norm = c2dimg_norm[crop_h:-crop_h, crop_w:-crop_w, :]
            dimg_norm_inp = dimg_norm[crop_h:-crop_h, crop_w:-crop_w]
            # 240, 320 > 200, 240
            prev_sh = np.shape(dimg_norm_inp)

            c2dimg_norm = cv2.resize(c2dimg_norm, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            dimg_norm_inp = cv2.resize(dimg_norm_inp, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

            input_RGBD = np.concatenate((c2dimg_norm, np.expand_dims(dimg_norm_inp, axis=-1)), axis=-1)
            input_RGBD_tensor = transforms.ToTensor()(input_RGBD)
            input_RGBD_tensor = torch.unsqueeze(input_RGBD_tensor, 0)
            input_RGBD_tensor = input_RGBD_tensor.to(device).type(torch.cuda.FloatTensor)

            with torch.no_grad():
                output = model.forward_test(input_RGBD_tensor)

            output = np.squeeze(output.cpu().numpy())

            output = cv2.resize(output, dsize=(prev_sh[1], prev_sh[0]), interpolation=cv2.INTER_NEAREST)
            dimg_norm[crop_h:-crop_h, crop_w:-crop_w] = output

            output = (output + 1) / 2.0
            cv2.imshow("inpainted depth", output)

            dimg_norm_out = (dimg_norm + 1) / 2.0
            cv2.imshow("final depth", dimg_norm_out)

            cv2.imshow("inp input rgb", c2dimg_norm)
            dimg_norm_vis = (dimg_norm_inp + 1) / 2.0
            cv2.imshow("inp input depth", dimg_norm_vis)

            dimg_norm_out = dimg_norm_out * d_max
            dimg_inp = np.copy(dimg_norm_out)
        # output value range : -1~1
        # need depth map with original range [d_min, d_max]

        model_img = hpf.run(dimg_inp)

        ## visualize
        cv2.imshow("c2d", np.uint8(c2dimg))
        cv2.moveWindow('c2d', 0, 200)

        dimg_norm_c[np.nonzero(dimg_norm_c)] -= 110
        dimg_norm_c *= (255. / np.max(dimg_norm_c))
        dimg_cmap = cv2.applyColorMap(np.uint8(dimg_norm_c), cv2.COLORMAP_JET)
        dimg3c = np.uint8(cv2.cvtColor(dimg_norm, cv2.COLOR_GRAY2BGR))
        dimg3c[dimg > 0] = [255, 0, 0]

        final_img = cv2.addWeighted(model_img, 1.0, dimg3c, 0.5, 0)
        cv2.imshow("dimg3c", np.uint8(dimg_cmap))
        cv2.imshow("model", final_img)

        cv2.moveWindow('dimg3c', hpf.width, 200)
        cv2.moveWindow('model', 3 * hpf.width, 200)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break




