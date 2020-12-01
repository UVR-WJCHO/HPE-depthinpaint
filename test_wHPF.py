from __future__ import division

import os
import sys
import argparse
import time

import cv2
import numpy as np

from DI_module import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='val', help='test')
    parser.add_argument('--name', type=str, default='_rgbd_crop', help='checkpoint name')
    parser.add_argument('--batch', type=int, default=1, help='size of the batches')
    opt = parser.parse_args()
    print(opt)

    dataset_type = 'obman'     ## 'obman', 'camera', 'self_image'

    flag_visualize = False
    flag_tracker = False

    if dataset_type == 'obman':
        dataloader_name = ImageDataset_obman
    elif dataset_type == 'self_image':
        dataloader_name = ImageDataset_self

    DI = DepthInpaint_module(cuda=True, target=opt.name, mode=opt.mode, batchSize=opt.batch, dataset=dataset_type)
    """
    if opt.mode == 'train':
        lr = 0.0001
        datasize = 20000
        set_viz = False
        set_OFE_eval = False
        DI._set_trainoption(lr, datasize, set_viz, set_OFE_eval)

        DI._set_memory()
        DI._set_module()
        DI._set_optimizer()

        if opt.load: DI._ckpfile_load()
        else: DI._ckpfile_init()

        DI._set_lossandlr()

        dataloader = DI._set_dataloader(dataloader_name)
        logger = DI._set_logger(dataloader)

        DI._create_output_pth()
        for epoch in range(DI.init_epoch, DI.n_epoch):
            DI._save_ckp(epoch)
            for i, batch in enumerate(dataloader):
                DI.train(i, batch)

            # Update learning rates
            DI.lr_scheduler_G.step()
            DI.lr_scheduler_D_B.step()

        sys.stdout.write('\n End training')
    """
    if opt.mode == 'test' or opt.mode == 'val':
        DI._set_memory()
        DI._set_module()
        DI._ckpfile_load()
        dataloader = DI._set_dataloader(dataloader_name)
        DI._create_output_pth()

        for i, batch in enumerate(dataloader):
            #### extract input ###
            A_tensor = batch['A']  # RGBD
            A_rgb_tensor = batch['rgb_A']

            ### run the depth inpaint module ###
            B_fake = DI.test(i, batch, A_tensor, A_rgb_tensor)

            #### visualize ###
            if flag_visualize:
                A = A_tensor.permute(0, 2, 3, 1)
                A = np.squeeze(A.cpu().float().numpy())
                A_rgb = A[:, :, :-1]
                A_rgb_norm = A_rgb * 255.0
                A_rgb_norm /= A_rgb_norm.max()
                cv2.imshow('A_rgb', A_rgb_norm)

                A_d = A[:, :, -1]
                A_d_norm = (A_d + 1.0) / 2.0
                A_d_norm /= A_d_norm.max()
                A_d_vis = np.uint8(np.squeeze(A_d_norm) * 255)
                A_d_vis = cv2.applyColorMap(A_d_vis, cv2.COLORMAP_JET)
                cv2.imshow('A_d', A_d_vis)

                B_fake = (B_fake + 1.0) / 2.0
                B_fake_vis = np.uint8(np.squeeze(B_fake) * 255)
                B_fake_vis = cv2.applyColorMap(B_fake_vis, cv2.COLORMAP_JET)
                cv2.imshow('fake_d', B_fake_vis)
                cv2.waitKey(0)

            #if flag_tracker:



            sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

        sys.stdout.write('\n End testing')
