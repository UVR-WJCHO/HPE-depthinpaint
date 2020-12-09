import os
import sys
import numpy as np
import cv2
import pylib as py
from pathlib import Path
import math
#from pylib import texturedSquare_PIL
import matplotlib as plt

import random
from PIL import Image, ImageOps


def _image_preprocessing(filename, xsize, ysize):
    im = cv2.imread(filename)
    sh = np.shape(im)
    x = im[:, 0:sh[0], 0]

    avg = np.mean(x)
    temp_x = np.array(x, dtype=np.float32)
    temp_x[x>0] -= avg
    temp_x /= 255.
    x = np.copy(temp_x)

    y_prev = im[:, sh[0]:, :]
    b, g, r = cv2.split(y_prev)

    temp_mask = np.zeros((sh[0], sh[0]))
    temp_mask[r > 130] = 1
    temp_mask[r < 130] = -1
    temp_mask[r < 10] = 0
    temp_x[r > 130] = 1

    y = np.zeros((sh[0], sh[0], 2))
    y[:, :, 0] = temp_x
    y[:, :, 1] = temp_mask

    return x[:,:,np.newaxis], y


def generate_random(mid):
    # mid = (h_x, w_x)
    random.seed()
    list_of_points = []

    # initialize parameter for generate polygon
    point_per_edge = [1, 2]
    h = random.uniform(25, 100)
    w = random.uniform(25, 100)

    if h + w < 70:
        h *= 2

    # upper side
    point_cnt = random.choice(point_per_edge)
    tmp_list = []
    p_h = int(mid[0] - h / 2)
    for t in range(point_cnt):
        p_w = int(random.uniform(mid[1] - w/2, mid[1] + w/2))
        tmp_list.append(p_w)

    if len(tmp_list) == 1:
        list_of_points.append([tmp_list[0], p_h])
    elif tmp_list[0] > tmp_list[1]:
        list_of_points.append([tmp_list[0], p_h])
        list_of_points.append([tmp_list[1], p_h])
    else:
        list_of_points.append([tmp_list[1], p_h])
        list_of_points.append([tmp_list[0], p_h])

    # left side
    point_cnt = random.choice(point_per_edge)
    tmp_list = []
    p_w = int(mid[1] - w / 2)
    for t in range(point_cnt):
        p_h = int(random.uniform(mid[0] - h / 2, mid[0] + h / 2))
        tmp_list.append(p_h)

    if len(tmp_list) == 1:
        list_of_points.append([p_w, tmp_list[0]])
    elif tmp_list[0] > tmp_list[1]:
        list_of_points.append([p_w, tmp_list[1]])
        list_of_points.append([p_w, tmp_list[0]])
    else:
        list_of_points.append([p_w, tmp_list[0]])
        list_of_points.append([p_w, tmp_list[1]])

    # lower side
    point_cnt = random.choice(point_per_edge)
    tmp_list = []
    p_h = int(mid[0] + h / 2)
    for t in range(point_cnt):
        p_w = int(random.uniform(mid[1] - w / 2, mid[1] + w / 2))
        tmp_list.append(p_w)
    if len(tmp_list) == 1:
        list_of_points.append([tmp_list[0], p_h])
    elif tmp_list[0] > tmp_list[1]:
        list_of_points.append([tmp_list[1], p_h])
        list_of_points.append([tmp_list[0], p_h])
    else:
        list_of_points.append([tmp_list[0], p_h])
        list_of_points.append([tmp_list[1], p_h])

    # right side
    point_cnt = random.choice(point_per_edge)
    tmp_list = []
    p_w = int(mid[1] + w / 2)
    for t in range(point_cnt):
        p_h = int(random.uniform(mid[0] - h / 2, mid[0] + h / 2))
        tmp_list.append(p_h)

    if len(tmp_list) == 1:
        list_of_points.append([p_w, tmp_list[0]])
    elif tmp_list[0] > tmp_list[1]:
        list_of_points.append([p_w, tmp_list[0]])
        list_of_points.append([p_w, tmp_list[1]])
    else:
        list_of_points.append([p_w, tmp_list[1]])
        list_of_points.append([p_w, tmp_list[0]])

    return list_of_points


def click(event, x, y, flags, param):
    global target
    if event == cv2.EVENT_LBUTTONDOWN:
        target = (x, y)
        cv2.circle(img, target, 3, (0, 0, 255), 1)
        cv2.imshow('Image', img)


def fit_image(image):
    #im = ImageOps.fit(image, size=(240, 240))

    im = image.resize(size=(256, 256))

    return np.array(im)

def normalizing_rgb(rgb):
    norm = np.zeros(np.shape(rgb), np.float32)

    b = rgb[:, :, 0]
    g = rgb[:, :, 1]
    r = rgb[:, :, 2]

    sum = b + g + r

    norm[:, :, 0] = b / sum * 255.0
    norm[:, :, 1] = g / sum * 255.0
    norm[:, :, 2] = r / sum * 255.0

    norm_rgb = cv2.convertScaleAbs(norm)

    return norm_rgb


def change_rgb_order(rgb):
    # loading with pillow change channel order
    B_rgb_temp = rgb.astype(float)
    B_rgb_temp[:, :, 0] = rgb[:, :, 2] / 255.
    B_rgb_temp[:, :, 2] = rgb[:, :, 0] / 255.
    B_rgb_temp[:, :, 1] = rgb[:, :, 1] / 255.

    return B_rgb_temp



if __name__ == '__main__':

    global img
    global target
    mode = 3

    if mode == 3:
        print("mode : ", mode)
        namelist_hand_color = []
        namelist_hand_depth = []

        namelist_obj_color = []
        namelist_obj_depth = []

        view_type = "exo"

        if view_type == "ego":
            for filename in Path(os.path.join('../datasets/self_2020/woOBJ/ego')).rglob('c2d/*.png'):
                namelist_hand_color.append(filename)
            for filename in Path(os.path.join('../datasets/self_2020/woOBJ/ego')).rglob('depth/*.png'):
                namelist_hand_depth.append(filename)
        elif view_type == "exo":
            for filename in Path(os.path.join('../datasets/self_2020/woOBJ/exo')).rglob('c2d/*.png'):
                namelist_hand_color.append(filename)
            for filename in Path(os.path.join('../datasets/self_2020/woOBJ/exo')).rglob('depth/*.png'):
                namelist_hand_depth.append(filename)



        for filename in Path(os.path.join('../../Object-Renderer-pyrender/examples/images/box')).rglob('color/*.png'):
            namelist_obj_color.append(filename)
        for filename in Path(os.path.join('../../Object-Renderer-pyrender/examples/images/box')).rglob('depth/*.png'):
            namelist_obj_depth.append(filename)
        for filename in Path(os.path.join('../../Object-Renderer-pyrender/examples/images/banana')).rglob('color/*.png'):
            namelist_obj_color.append(filename)
        for filename in Path(os.path.join('../../Object-Renderer-pyrender/examples/images/banana')).rglob('depth/*.png'):
            namelist_obj_depth.append(filename)

        tmp = len(namelist_hand_depth)
        len_obj = len(namelist_obj_depth)
        len_dataset = len(namelist_hand_depth) * 5

        # load texture data
        texture_list = []
        for filename in Path(os.path.join('texture')).rglob('*.png'):
            texture_list.append(filename)
        for filename in Path(os.path.join('texture')).rglob('*.jpg'):
            texture_list.append(filename)
        len_texture = len(texture_list)

        # main process
        key_flag = True

        #pair_A_RGBD = np.zeros((len_dataset, 256, 256, 4))
        #pair_B_DM = np.zeros((len_dataset, 256, 256, 2))

        save_cnt = 0
        itr = 0

        p_total = np.random.permutation(len_dataset)
        test_len = int(len_dataset / 10)
        train_len = len_dataset - test_len

        p_obj = np.random.permutation(len_obj)
        p_tex = np.random.permutation(len_texture)

        for i in range(len_dataset):
            if i == len_dataset - 1:
                # print("saving...")
                # name_1 = 'pair_A_RGBD_10k_1.npy'
                # name_2 = 'pair_B_DM_10k_1.npy'
                #
                # np.save(name_1, pair_A_RGBD)
                # np.save(name_2, pair_B_DM)
                print("done")

            idx = p_total[i]

            hand_c = np.array(Image.open(str(namelist_hand_color[idx % tmp])))
            hand_d = np.array(Image.open(str(namelist_hand_depth[idx % tmp])))


            idx_obj = p_obj[idx % len_obj]
            idx_tex = p_tex[idx % len_texture]
            obj_c = np.array(Image.open(str(namelist_obj_color[idx_obj])))
            obj_d = np.array(Image.open(str(namelist_obj_depth[idx_obj])))
            tex = np.array(Image.open(str(texture_list[idx_tex])))
            tex = cv2.resize(tex, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

            if len(hand_c[0, 0, :]) == 4:
                hand_c = hand_c[:, :, :-1]
                hand_d = hand_d[:, :, 1]
                hand_d[hand_d==1] = 0


            if view_type == "exo":
                hand_d[hand_d>600] = 0
                hand_c[hand_d == 0, :] = [0, 0, 0]

                hand_d = hand_d[40:-40, 40:-40]
                hand_c = hand_c[40:-40, 40:-40, :]


            hand_min_d = np.min(np.nonzero(hand_d))
            obj_min_d = np.min(np.nonzero(obj_d))

            hand_max_d = np.max(hand_d)
            obj_max_d = np.max(obj_d)


            """
            + depth : closer ~ smaller
            
            1. find median depth of each hand/obj
            2. align to value closer to 128
            3. for each pixel, assign fuzed value by the depth
            3.1 assign closer depth, closer rgb
            
            """
            hand_d = hand_d.astype(np.int16)
            obj_d = obj_d.astype(np.int16)

            hand_mid = (int(hand_max_d) + int(hand_min_d)) / 2
            obj_mid = (int(obj_max_d) + int(obj_min_d)) / 2

            # hand_off = abs(128 - hand_mid)
            # obj_off = abs(128 - obj_mid)
            # if obj_off < hand_off:
            #     offset = hand_mid - obj_mid
            #     hand_d[np.nonzero(hand_d)] -= offset
            # elif hand_off < obj_off:
            #     offset = obj_mid - hand_mid
            #     obj_d[np.nonzero(obj_d)] -= offset

            hand_off = int(hand_mid - 128) + np.random.randint(-10, 10)
            obj_off = int(obj_mid - 128) + np.random.randint(-10, 10)
            hand_d[np.nonzero(hand_d)] -= hand_off
            obj_d[np.nonzero(obj_d)] -= obj_off

            hand_d = cv2.resize(hand_d, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            obj_d = cv2.resize(obj_d, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            hand_c = cv2.resize(hand_c, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

            both_c = np.zeros((256, 256, 3), dtype=np.uint8)
            both_d = np.zeros((256, 256), dtype=np.uint8)
            mask_obj = np.zeros((256, 256), dtype=np.uint8)
            for k in range(256):
                for j in range(256):
                    h_d = hand_d[k, j]
                    o_d = obj_d[k, j]

                    if h_d < o_d and h_d != 0:
                        both_d[k, j] = h_d
                        both_c[k, j, 0] = hand_c[k, j, 0]
                        both_c[k, j, 1] = hand_c[k, j, 1]
                        both_c[k, j, 2] = hand_c[k, j, 2]
                    elif o_d < h_d and o_d != 0:
                        both_d[k, j] = o_d
                        both_c[k, j, 0] = tex[k, j, 0]
                        both_c[k, j, 1] = tex[k, j, 1]
                        both_c[k, j, 2] = tex[k, j, 2]
                        mask_obj[k, j] = 255
                    elif o_d > 0:
                        both_d[k, j] = o_d
                        both_c[k, j, 0] = tex[k, j, 0]
                        both_c[k, j, 1] = tex[k, j, 1]
                        both_c[k, j, 2] = tex[k, j, 2]
                        mask_obj[k, j] = 255
                    else:
                        both_d[k, j] = h_d
                        both_c[k, j, 0] = hand_c[k, j, 0]
                        both_c[k, j, 1] = hand_c[k, j, 1]
                        both_c[k, j, 2] = hand_c[k, j, 2]
            """
            cv2.imshow("hand_c", hand_c)
            cv2.imshow("hand_d", np.uint8(hand_d))
            cv2.imshow("obj_c", obj_c)
            cv2.imshow("obj_d", np.uint8(obj_d))
            cv2.imshow("both_c", both_c)
            cv2.imshow("both_d", both_d)
            cv2.imshow("mask_obj", mask_obj)
            cv2.waitKey(0)
            """
            masked_c = both_c.copy()
            masked_c[mask_obj == 0] = 0
            # cv2.imshow("masked_c", masked_c)
            # cv2.waitKey(0)
            """
            save as image
            """
            if i < train_len:
                name_base = "../datasets/self_2020/wOBJ/train/" + view_type
            else:
                name_base = "../datasets/self_2020/wOBJ/test/" + view_type

            name_Ac = name_base + "/A_color/" + str(save_cnt) + ".png"
            cv2.imwrite(name_Ac, both_c)
            name_Ad = name_base + "/A_depth/" + str(save_cnt) + ".png"
            cv2.imwrite(name_Ad, both_d)
            name_Bd = name_base + "/B_depth/" + str(save_cnt) + ".png"
            cv2.imwrite(name_Bd, hand_d)
            name_Bm = name_base + "/B_mask/" + str(save_cnt) + ".png"
            cv2.imwrite(name_Bm, mask_obj)

            name_Bmc = name_base + "/A_masked_color/" + str(save_cnt) + ".png"
            cv2.imwrite(name_Bmc, masked_c)

            # A = np.concatenate((both_c, np.expand_dims(both_d, axis=-1)), axis=-1)
            # B = np.concatenate((np.expand_dims(hand_d, axis=-1), np.expand_dims(mask_obj, axis=-1)), axis=-1)
            #
            # pair_A_RGBD[save_cnt] = A
            # pair_B_DM[save_cnt] = B

            save_cnt += 1
            key = cv2.waitKey(1)

            if i % 100 == 0:
                msg = view_type + " itr in total --- "
                print(msg, (i, len_dataset, p_total[i]))

            if key == ord('q') or key == 27:
                break
