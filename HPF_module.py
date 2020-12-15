import os, time
import numpy as np
import matplotlib.pyplot as plt

import ctypes
import _ctypes
from ctypes import cdll

import sys
sys.path.append('./HPF')

import pickle

import cv2  
from Hmf_structure import configureState_hand

from pyquaternion import Quaternion
import csv

from Sensor import Sensor
from Renderer import Renderer
from Renderer import GlCamera
from HandTracker import HandTracker

import matplotlib as mplt
import PIL.Image as pl
import matplotlib.pyplot as plt



def getCsvData(filename):
    csv_file=open(filename,'r')
    csv_reader=csv.reader(csv_file)
    label_csv=[]

    for row in csv_reader:    
        label_csv.append(row)           
    csv_file.close()

        
    outdata=np.asarray(label_csv,dtype='float')

    return outdata


class HPF_module:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # file path
        sensor_dllpath = './HPF/c++/bin/GYSensorReader.dll'
        render_dllpath = './HPF/c++/bin/GYHandTracker.dll'

        # sensor
        cameratype = b'playcamera'  # b'realcamera'  b'playcamera'
        dataset = b'DexterHO'  # b'EgoDexter' b'DexterHO' b'Self'
        seq_name = b'Occlusion'
        imagepath = b'./datasets/' + dataset + b'/' + seq_name + b'/'

        fx = 477.9 / 2.0
        fy = 477.9 / 2.0
        znear = 0.01
        zfar = 1000.0
        gyroSamplingTime = 0.01

        if dataset == b'EgoDexter' or dataset == b'Self' or cameratype == b'realcamera':
            cx = 640.0 / 2.0
            cy = 480.0 / 2.0
            width = int(640)
            height = int(480)
        else:
            cx = 320.0 / 2.0
            cy = 240.0 / 2.0
            width = int(320)
            height = int(240)

        useGyro = False

        self.renderer = Renderer(cameratype, render_dllpath, width, height)
        self.sensor = Sensor(sensor_dllpath, fx, fy, width, height, znear, zfar, cx, cy,
                        cameratype, imagepath, gyroSamplingTime, useGyro, dataset)

        self.sensor.init()

        # tracker
        # stdevBag = [0.08, 0.08, 0.08]  # paper: (0.06 , 0.06 , 0.06)
        stdevBag = [0.06, 0.06, 0.06]
        gyro_alpha = 0.4  # paper:0.4
        # covBag = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.09]  # paper: 0.4,0.4,0.4,0.4,0.4,0.4,0.06
        covBag = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.06]

        self.handtracker = HandTracker(self.renderer, "..", stdevBag, gyro_alpha, covBag)


        self.handPose, self.frame = self._init_pose(cameratype, dataset, seq_name)

        self.hand_q = Quaternion(self.handPose[3], self.handPose[4], self.handPose[5], self.handPose[6])
        self.width = width
        self.height = height


    def _init_pose(self, cameratype, dataset, seq_name):
        # initial solution
        init_frame = 0
        initialPose = [20, 76., 400,
                       0.022, 0.047, 0.437, 0.897,  # w,x,y,z
                       0., 0., 0., 0.,
                       0., -12., 0., 0.,
                       0., -2.5, 0., 0.,
                       0., 2.47, 0., 0.,
                       0., -5., 0., 0.]

        if cameratype == b'playcamera' and dataset == b'DexterHO' and seq_name == b'Rotate':
            init_frame = 1
            initialPose = [-82., 54.3, 467.9,
                           0.123, -0.295, 0.526, 0.788,
                           -20.4, 18.99, -4.74, -3.86,
                           -25.29, -0.861, -20., -1.282,
                           -0.274, 4.297, -5.5, -28.7,
                           -0.814, 9.76, -2.439, -1.22,
                           -0.961, -4.499, -3.157, -41.9]

        if cameratype == b'playcamera' and dataset == b'DexterHO' and seq_name == b'Occlusion':
            init_frame = 1
            initialPose = [25., 61, 473.6,
                           0.014, 0.062, 0.567, 0.821,
                           5.92, 22.9, -0.61, -1.65,
                           -8.15, -4.0, -1.88, -0.513,
                           -1.30, 3.74, -1.75, -14.2,
                           -0.693, 6.69, -1.72, -11.73,
                           -4.23, -3.98, -10.63, -47.89]

        if cameratype == b'playcamera' and dataset == b'Self':
            init_frame = 1
            initialPose = [25., 61, 473.6,
                           0.014, 0.062, 0.567, 0.821,
                           5.92, 22.9, -0.61, -1.65,
                           -8.15, -4.0, -1.88, -0.513,
                           -1.30, 3.74, -1.75, -14.2,
                           -0.693, 6.69, -1.72, -11.73,
                           -4.23, -3.98, -10.63, -47.89]

        initialPose = np.asarray(initialPose, 'float32')

        return initialPose, init_frame

    def _get_input(self):
        self.sensor.setFrame(self.frame)
        if self.sensor.runSensor() == False:
            print("sensor doesn't work")
            return
        cimg, dimg_orig, c2dimg = self.sensor.getImages()

        return dimg_orig, c2dimg

    def run(self, dimg):
        # debugging local
        debugBag_local = {}
        debugBag_local['particles'] = []
        debugBag_local['particles_resampled'] = []
        debugBag_local['renderedParticles'] = []
        debugBag_local['renderedParticles_resampled'] = []
        debugBag_local['likelihood'] = []
        debugBag_local['resampleCount'] = []
        debugBag_local['joint2D'] = []
        debugBag_local['joint3D'] = []


        # -----------model-based tracking--------------------------#

        self.renderer.transferObservation2GPU(dimg)

        self.handPose = self.handtracker.run_original(self.handPose, debugBag_local)
        self.hand_q = Quaternion(self.handPose[3], self.handPose[4], self.handPose[5], self.handPose[6])

        # -----------render tracking result--------------------------#
        handPose_render = np.copy(self.handPose)
        self.renderer.render(handPose_render, b'depth')
        model_img = self.renderer.getResizedDepthTexture(self.width, self.height)
        cv2.normalize(model_img, model_img, 0, 255, cv2.NORM_MINMAX)
        model_img = cv2.applyColorMap(np.uint8(model_img), 5)

        # end
        self.frame += 1

        return model_img


if __name__=="__main__":

    hpf = HPF_module()

    while(1):
        dimg, c2dimg = hpf._get_input()

        cv2.imshow("c2d", np.uint8(c2dimg))
        cv2.moveWindow('c2d', 0, 200)

        hpf.run(dimg)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break




