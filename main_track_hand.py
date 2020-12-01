import os, time
import numpy as np
import matplotlib.pyplot as plt

import ctypes
import _ctypes
from ctypes import cdll

import sys

import pickle

import cv2  
from Hmf_structure import configureState_hand

from pyquaternion import Quaternion
import csv

from Sensor import Sensor
from Renderer import Renderer
from Renderer import GlCamera
from HandTracker import HandTracker

from pix2pix_net.pix2pix_fin import Pix2pix
import tensorflow as tf
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


if __name__=="__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    methodNumber=2
    
    #method 1 : baseline (original HMF)
    if methodNumber==1:
        useGyro=False
        trackingType="original"
        flag_activ_track = True
        flag_activ_net = True
        flag_save = False
        flag_re = False

        flag_saveImg = False

    if methodNumber==2:
        useGyro=False
        trackingType="original"
        flag_activ_track = False
        flag_activ_net = False
        flag_save = False
        flag_re = False

        flag_saveImg = True

    allPose_txt = open('Dexter_Occlusion_pose_in.txt', 'w')
    #file path
    sensor_dllpath = 'D:/Research_2020/HPF_handtracker/c++/bin/GYSensorReader.dll'
    render_dllpath = 'D:/Research_2020/HPF_handtracker/c++/bin/GYHandTracker.dll'
    result_path = './experiment/'
    """
        For b'EgoDexter'
        Desk Fruits Kitchen Rotunda

        For b'DexterHO'
        Grasp1 Grasp2 Occlusion Pinch Rigid Rotate
        """

    exp_name = '45epo'
    checkpoint_path = './pix2pix_model/' + exp_name
    print("checkpoint path : ", checkpoint_path)

    # sensor
    cameratype = b'realcamera'  # b'realcamera'  b'playcamera'
    cameraname = b'SR300'  # b'SR300' OR  b'D435'

    dataset = b'DexterHO'  # b'EgoDexter' b'DexterHO' b'Self'
    seq_name = b'Occlusion'
    imagepath = b'D:/Research_2020/HPF_handtracker/dataset/' + dataset + b'/' + seq_name + b'/'

    if dataset == b'Self':
        imagepath = b'D:/Research_2020/HPF_handtracker/dataset/' + dataset + b'/'

    time_delay=0.01
    
    #configuration
    if cameraname==b'SR300':
        fx=477.9/2.0
        fy=477.9/2.0
    if cameraname==b'D435':
        fx=386.36/2.0
        fy=386.36/2.0

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

    znear=0.01
    zfar=1000.0
    gyroSamplingTime=0.01
    
    handImgSize=256
    
    calibMat=np.matrix([[fx,0,cx],[0,fy,cy],[0,0,1]])
    print('cameratype:',cameratype,'usegyro:',useGyro)
        
    #renderer
    if not 'renderer' in locals():
        renderer=Renderer(cameratype,render_dllpath,width,height)

    #sensor
    if not 'sensor' in locals() and cameratype!=b'glcamera':
        sensor=Sensor(sensor_dllpath,fx,fy,width,height,znear,zfar,cx,cy,
                       cameratype,imagepath,gyroSamplingTime,useGyro, dataset)

    sensor.init()
    
    #tracker    
    #stdevBag = [0.08, 0.08, 0.08]  # paper: (0.06 , 0.06 , 0.06)
    stdevBag = [0.06, 0.06, 0.06]
    gyro_alpha = 0.4  # paper:0.4
    #covBag = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.09]  # paper: 0.4,0.4,0.4,0.4,0.4,0.4,0.06
    covBag = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.06]
    handtracker=HandTracker(renderer,"..",stdevBag,gyro_alpha,covBag) 
    stateNum=7

    
    #initial solution
    init_frame = 0
    initialPose=[20,76.,400,
                0.022,0.047,0.437,0.897, #w,x,y,z
                0.,0.,0.,0.,
                0.,-12.,0.,0.,
                0.,-2.5,0.,0.,
                0.,2.47,0.,0.,
                0.,-5.,0.,0.]
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
    
    initialPose=np.asarray(initialPose,'float32')
    handPose=np.copy(initialPose)
    hand_q=Quaternion(handPose[3],handPose[4],handPose[5],handPose[6])

    #debugging global
    debugBag={}
    debugBag['debug_frame']=[]
    debugBag['debug_depthimg']={}
    debugBag['debug_handPose']={}
    debugBag['debug_hand_q_init']={}
    debugBag['debug_particles']={}
    debugBag['debug_particles_resampled']={}
    debugBag['debug_renderedParticles']={}
    debugBag['debug_renderedParticles_resampled']={}
    debugBag['debug_cameraTime']={}
    
    debugBag['debug_likelihood']={}
    debugBag['debug_resampleCount']={}
    for i in range(stateNum):
        debugBag['debug_likelihood'][i]={}
        debugBag['debug_resampleCount'][i]={}
    debugBag['debug_joint2D']={}
    debugBag['debug_joint3D'] = {}
    #---------start---------------#
    frame=init_frame
    processing_time=0

    cmap_value = 110

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    if flag_activ_net:
        with tf.device('/gpu:1'):
            model = Pix2pix()
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count={'GPU': 1})
            sess = tf.Session(config=config)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))



    while(1):
        key = cv2.waitKey(1)

        #debugging local
        debugBag_local={}
        if cameratype!=b'realcamera':
            debugBag_local['particles']=[]
            debugBag_local['particles_resampled']=[]
            debugBag_local['renderedParticles']=[]
            debugBag_local['renderedParticles_resampled']=[]
            debugBag_local['likelihood']=[]
            debugBag_local['resampleCount']=[]
            debugBag_local['joint2D']=[]
            debugBag_local['joint3D'] = []

        #----------camera-----------------------------#
        #get camera image and time
        sensor.setFrame(frame)
        if sensor.runSensor()==False:
            print("sensor doesn't work")
            break

        cimg, dimg_orig, c2dimg = sensor.getImages() #, cameraTime
        #cv2.imshow("original", np.uint8(cimg))
        #cv2.moveWindow('original', 0, 200)
        """
        cv2.putText(c2dimg, str(frame),bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        """
        c2dimg_png = np.uint8(c2dimg)
        cv2.imshow("c2d", c2dimg_png)
        cv2.moveWindow('c2d', 0, 200)

        if flag_saveImg:
            folder = 'trial_0'
            c2d_name = 'data/self_2020/' + folder + '/c2d/c2d_' + str(frame) + '.png'
            depth_name = 'data/self_2020/' + folder + '/depth/depth_' + str(frame) + '.png'

            dimg_png = np.copy(dimg_orig)
            nonzero_arr = np.nonzero(dimg_png)
            if not len(nonzero_arr[0]) == 0:
                min_v = np.min(dimg_png[nonzero_arr])
                max_v = np.max(dimg_png)
                gap = max_v - 256 + 20
                dimg_png[nonzero_arr] -= gap
                dimg_png[dimg_png < 0] = 0
                dimg_png = np.uint8(dimg_png)
                cv2.imshow("dimg_png", dimg_png)


                mplt.pyplot.imsave(c2d_name, c2dimg_png)
                mplt.pyplot.imsave(depth_name, dimg_png)


        dimg = np.copy(dimg_orig)
        if cameratype==b'playcamera' and dataset == b'Self':
            dimg[dimg > 500] = 0
        else:
            dimg[dimg > 600] = 0

        dimg_norm = 255 * (dimg - dimg.min()) / (dimg.max() - dimg.min())
        dimg_norm_c = np.copy(dimg_norm)

        dimg_norm_c[np.nonzero(dimg_norm_c)] -= cmap_value
        dimg_norm_c *= (255. / np.max(dimg_norm_c))
        #print("dimg_norm min, max : ", np.min(np.nonzero(dimg_norm)), np.max(dimg_norm))
        dimg3c = np.uint8(cv2.cvtColor(dimg_norm, cv2.COLOR_GRAY2BGR))
        #cv2.imshow("dimg_norm", np.uint8(dimg_norm))
        dimg_cmap = cv2.applyColorMap(np.uint8(dimg_norm_c), cv2.COLORMAP_JET)
        cv2.imshow("dimg3c", np.uint8(dimg_cmap))
        cv2.moveWindow('dimg3c', width, 200)
        #---------segment hand from the camera image-------------#
        # sensor.findHand() #0.002 sec
        # dimg,dimg_norm,dimg3c=sensor.segmentHand(dimg,dimg_norm)
        dimg3c_blue=dimg3c.copy()
        dimg3c_blue[dimg>0]=[255,0,0]

        if flag_activ_net:
            net_color = np.float32(c2dimg)
            net_depth = np.float32(dimg_norm)

            a_hsv = mplt.colors.rgb_to_hsv(net_color)
            a_hsv[:, :, 2] = 0.5
            net_color = mplt.colors.hsv_to_rgb(a_hsv) * 2.0 - 1.0

            # crop hand region. resize to (256, 256)
            if cameratype==b'playcamera' and dataset == b'DexterHO':
                net_color = net_color[:, 40:-40]
                net_depth = net_depth[:, 40:-40]
            elif cameratype==b'realcamera': # 640 480
                net_color = net_color[:, 80:-80]
                net_depth = net_depth[:, 80:-80]

            # print("nonzero min : ", np.max(net_depth), np.min(net_depth[np.nonzero(net_depth)]))
            net_depth = net_depth * 2. / 255. - 1

            net_color = cv2.resize(net_color, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            net_depth = cv2.resize(net_depth, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

            # make it to (, , 4)
            net_in = np.concatenate([net_color, np.expand_dims(net_depth, axis=-1)], axis=-1)

            if flag_re:
                while True:
                    net_out = np.squeeze(model.sample_generator(sess, np.expand_dims(net_in, axis=0), is_training=False))
                    net_out = np.squeeze(np.array((net_out + 1) * 255. / 2.0, dtype='u1'))

                    # allow only inpainting on occluded area
                    net_out[net_depth == -1] = 0

                    # transport output to cropped region of original depth
                    if cameratype==b'playcamera' and dataset == b'DexterHO':
                        net_out = cv2.resize(net_out, dsize=(240, 240), interpolation=cv2.INTER_NEAREST)
                        net_out3c = np.uint8(cv2.cvtColor(net_out, cv2.COLOR_GRAY2BGR))
                        dimg_norm[:, 40:-40] = net_out
                        dimg = dimg.min() + (dimg_norm * (dimg.max() - dimg.min()) / 255.)
                    elif cameratype == b'realcamera':  # 640 480
                        net_out = cv2.resize(net_out, dsize=(480, 480), interpolation=cv2.INTER_NEAREST)
                        net_out3c = np.uint8(cv2.cvtColor(net_out, cv2.COLOR_GRAY2BGR))
                        dimg_norm[:, 80:-80] = net_out
                        dimg = dimg.min() + (dimg_norm * (dimg.max() - dimg.min()) / 255.)

                    cv2.imshow("inpainted", net_out3c)
                    key_iter = cv2.waitKey(0)
                    # a : re inpaint, d : save and break
                    if key_iter == ord('a'):
                        print("re")
                    elif key_iter == ord('s'):
                        fig_name = './data/DexterHO/Occlusion/' + str(frame) + '.png'
                        dimg_cmap = cv2.applyColorMap(np.uint8(dimg_norm), cv2.COLORMAP_JET)
                        cv2.imshow("inpainted_cmap", dimg_cmap)
                        pl.fromarray(np.uint8(dimg_cmap)).save(fig_name)
                        break

            else:
                net_out = np.squeeze(model.sample_generator(sess, np.expand_dims(net_in, axis=0), is_training=False))
                net_out = np.squeeze(np.array((net_out + 1) * 255. / 2.0, dtype='u1'))

                # allow only inpainting on occluded area
                net_out[net_depth == -1] = 0

                # transport output to cropped region of original depth
                if cameratype == b'playcamera' and dataset == b'DexterHO':
                    net_out = cv2.resize(net_out, dsize=(240, 240), interpolation=cv2.INTER_NEAREST)
                    net_out3c = np.uint8(cv2.cvtColor(net_out, cv2.COLOR_GRAY2BGR))

                    dimg_norm[:, 40:-40] = net_out
                    dimg = dimg.min() + (dimg_norm * (dimg.max() - dimg.min()) / 255.)
                elif cameratype == b'realcamera':  # 640 480
                    net_out = cv2.resize(net_out, dsize=(480, 480), interpolation=cv2.INTER_NEAREST)
                    net_out3c = np.uint8(cv2.cvtColor(net_out, cv2.COLOR_GRAY2BGR))

                    dimg_norm[:, 80:-80] = net_out
                    dimg = dimg.min() + (dimg_norm * (dimg.max() - dimg.min()) / 255.)

                #cv2.imshow("inpainted", net_out3c)
                dimg_norm_c = np.copy(dimg_norm)
                dimg_norm_c[np.nonzero(dimg_norm_c)] -= cmap_value
                dimg_norm_c *= (255. / np.max(dimg_norm_c))

                dimg_cmap = cv2.applyColorMap(np.uint8(dimg_norm_c), cv2.COLORMAP_JET)
                cv2.imshow("inpainted_cmap", dimg_cmap)

            #cv2.moveWindow('inpainted', 840, 200)
            cv2.moveWindow('inpainted_cmap', 2*width, 200)

        #-----------model-based tracking--------------------------#
        if flag_activ_track:
            renderer.transferObservation2GPU(dimg)

            #TIME=time.time()\
            if trackingType=="original":
                #start = time.time()
                handPose=handtracker.run_original(handPose,debugBag_local)
                #print("time : ", time.time() - start)
                
                # handpose를 frame마다 txt line으로 저장.
                for pose in handPose:
                    allPose_txt.write(str(pose)+' ')
                allPose_txt.write('\n')
                
            hand_q=Quaternion(handPose[3],handPose[4],handPose[5],handPose[6])

            if flag_save:
                for i in range(5):
                    jp3d = renderer.getJointPosition(i, 2)
                    debugBag_local['joint3D'].append(float(jp3d[0]))
                    debugBag_local['joint3D'].append(-float(jp3d[1]))
                    debugBag_local['joint3D'].append(float(jp3d[2]))
                    # print('jp3d',frame,i,jp3d)
                    jp2d = calibMat * np.reshape(jp3d, (3, 1))
                    jp2d /= jp2d[2]
                    debugBag_local['joint2D'].append(float(jp2d[0]))
                    debugBag_local['joint2D'].append(float(jp2d[1]))

            #-----------render tracking result--------------------------#
            handPose_render=np.copy(handPose)
            renderer.render(handPose_render,b'depth')
            model_img=renderer.getResizedDepthTexture(width,height)
            cv2.normalize(model_img,model_img,0,255,cv2.NORM_MINMAX)
            model_img=cv2.applyColorMap(np.uint8(model_img),5)
            #cv2.imshow("model1",model_img)
            final_img=cv2.addWeighted(model_img,1.0,dimg3c_blue,0.5,0)
            #final_img=cv2.addWeighted(model_img,1.0,dimg3c_orig,0.5,0)
            final_img=cv2.resize(final_img,(width,height))
            cv2.imshow("model",final_img)
            cv2.moveWindow('model', 3 * width, 200)
            if flag_save:
                # if useGyro==True:
                debugBag['debug_frame'].append(frame)
                debugBag['debug_particles'][frame] = debugBag_local['particles']
                debugBag['debug_particles_resampled'][frame] = debugBag_local['particles_resampled']
                debugBag['debug_renderedParticles'][frame] = debugBag_local['renderedParticles']
                debugBag['debug_renderedParticles_resampled'][frame] = debugBag_local['renderedParticles_resampled']
                debugBag['debug_likelihood'][frame] = debugBag_local['likelihood']
                debugBag['debug_resampleCount'][frame] = debugBag_local['resampleCount']
                debugBag['debug_joint2D'][frame] = debugBag_local['joint2D']
                debugBag['debug_joint3D'][frame] = debugBag_local['joint3D']


        if key==ord('q'):
            break

        cv2.waitKey(1)
        
        #end
        frame+=1

    print("processed frame : ", frame)

    if flag_save:
        """
        csv_path = result_path + 'position2d_' + '%s_' % dataset + '%s_' % seq_name + '%s' % exp_name + '_bare.csv'
        with open(csv_path, 'w', newline='') as f:
            wr = csv.writer(f)
            for fr in range(init_frame, frame - 1):
                wr.writerow(debugBag['debug_joint2D'][fr])
            f.close()
        """
        csv_path = result_path + 'position3d_' + '%s_' % dataset + '%s_' % seq_name + '%s' % exp_name + '_all.csv'
        with open(csv_path, 'w', newline='') as f:
            wr = csv.writer(f)
            for fr in range(init_frame, frame - 1):
                wr.writerow(debugBag['debug_joint3D'][fr])
            f.close()


        print("save position... frame:", frame - 1)
