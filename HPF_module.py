import ctypes
import _ctypes
from ctypes import cdll

import os, time, sys
import pickle
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from HPF.Hmf_structure import configureState_hand
from HPF.Sensor import Sensor
from HPF.Renderer import Renderer, GlCamera
from HPF.HandTracker import HandTracker


def getCsvData(filename):
    csv_file = open(filename, 'r')
    csv_reader = csv.reader(csv_file)
    label_csv = []

    for row in csv_reader:
        label_csv.append(row)
    csv_file.close()

    outdata = np.asarray(label_csv, dtype='float')

    return outdata

class HPF_module:
    def __init__(self):
        useGyro = False
        trackingType = "original"

        sensor_dllpath = 'D:/Research_2020/PyTorch-CycleGAN/HPF/c++/bin/GYSensorReader.dll'
        render_dllpath = 'D:/Research_2020/PyTorch-CycleGAN/HPF/c++/bin/GYHandTracker.dll'
        result_path = './experiment/'

        # sensor
        cameratype = b'playcamera'  # b'realcamera'  b'playcamera'
        cameraname = b'SR300'  # b'SR300' OR  b'D435'

        dataset = b'DexterHO'  # b'EgoDexter' b'DexterHO' b'Self'
        seq_name = b'Occlusion'
        imagepath = b'D:/Research_2020/HPF_handtracker/dataset/' + dataset + b'/' + seq_name + b'/'

        # SR300 camera configuration
        fx = 477.9 / 2.0
        fy = 477.9 / 2.0

        self._set_image_config(dataset, cameratype)
        self._set_initial_pose(dataset, cameratype, seq_name)

        znear = 0.01
        zfar = 1000.0
        gyroSamplingTime = 0.01
        handImgSize = 256

        self.renderer = Renderer(cameratype, render_dllpath, self.width, self.height)

        self.sensor = Sensor(sensor_dllpath, fx, fy, self.width, self.height, znear, zfar, self.cx, self.cy,
                            cameratype, imagepath, gyroSamplingTime, useGyro, dataset)

        # tracker setting
        stdevBag = [0.06, 0.06, 0.06]   # paper: (0.06 , 0.06 , 0.06)
        gyro_alpha = 0.4  # paper:0.4
        covBag = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.06]
        self.handtracker = HandTracker(self.renderer, "..", stdevBag, gyro_alpha, covBag)
        stateNum = 7

        # initial solution
        initialPose = np.asarray(self.initialPose, 'float32')
        self.handPose = np.copy(initialPose)

    def run(self, dimg):

        dimg_norm = 255 * (dimg - dimg.min()) / (dimg.max() - dimg.min())
        dimg3c = np.uint8(cv2.cvtColor(dimg_norm, cv2.COLOR_GRAY2BGR))
        dimg3c_blue = dimg3c.copy()
        dimg3c_blue[dimg > 0] = [255, 0, 0]

        self.renderer.transferObservation2GPU(dimg)
        handPose = self.handtracker.run_original(self.handPose)
        hand_q = Quaternion(handPose[3], handPose[4], handPose[5], handPose[6])

        # -----------render tracking result--------------------------#
        handPose_render = np.copy(handPose)
        self.renderer.render(handPose_render, b'depth')
        model_img = self.renderer.getResizedDepthTexture(self.width, self.height)
        cv2.normalize(model_img, model_img, 0, 255, cv2.NORM_MINMAX)
        model_img = cv2.applyColorMap(np.uint8(model_img), 5)
        # cv2.imshow("model1",model_img)
        final_img = cv2.addWeighted(model_img, 1.0, dimg3c_blue, 0.5, 0)
        # final_img=cv2.addWeighted(model_img,1.0,dimg3c_orig,0.5,0)
        final_img = cv2.resize(final_img, (self.width, self.height))
        cv2.imshow("model", final_img)

    def _set_initial_pose(self, dataset, cameratype, seq_name):
        self.init_frame = 0
        self.initialPose = [20, 76., 400,
                       0.022, 0.047, 0.437, 0.897,  # w,x,y,z
                       0., 0., 0., 0.,
                       0., -12., 0., 0.,
                       0., -2.5, 0., 0.,
                       0., 2.47, 0., 0.,
                       0., -5., 0., 0.]

        if cameratype == b'playcamera' and dataset == b'DexterHO' and seq_name == b'Rotate':
            self.init_frame = 1
            self.initialPose = [-82., 54.3, 467.9,
                           0.123, -0.295, 0.526, 0.788,
                           -20.4, 18.99, -4.74, -3.86,
                           -25.29, -0.861, -20., -1.282,
                           -0.274, 4.297, -5.5, -28.7,
                           -0.814, 9.76, -2.439, -1.22,
                           -0.961, -4.499, -3.157, -41.9]

        if cameratype == b'playcamera' and dataset == b'DexterHO' and seq_name == b'Occlusion':
            self.init_frame = 1
            self.initialPose = [25., 61, 473.6,
                           0.014, 0.062, 0.567, 0.821,
                           5.92, 22.9, -0.61, -1.65,
                           -8.15, -4.0, -1.88, -0.513,
                           -1.30, 3.74, -1.75, -14.2,
                           -0.693, 6.69, -1.72, -11.73,
                           -4.23, -3.98, -10.63, -47.89]

    def _set_image_config(self, dataset, cameratype):
        if dataset == b'EgoDexter' or dataset == b'Self' or cameratype == b'realcamera':
            self.cx = 640.0 / 2.0
            self.cy = 480.0 / 2.0
            self.width = int(640)
            self.height = int(480)
        else:
            self.cx = 320.0 / 2.0
            self.cy = 240.0 / 2.0
            self.width = int(320)
            self.height = int(240)
