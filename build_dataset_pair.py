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

        for filename in Path(os.path.join('datasets/self_2020/trial_0')).rglob('c2d/*.png'):
            namelist_hand_color.append(filename)
        for filename in Path(os.path.join('datasets/self_2020/trial_0')).rglob('depth/*.png'):
            namelist_hand_depth.append(filename)

        for filename in Path(os.path.join('../Object-Renderer-pyrender/examples/images/banana')).rglob('color/*.png'):
            namelist_obj_color.append(filename)
        for filename in Path(os.path.join('../Object-Renderer-pyrender/examples/images/banana')).rglob('depth/*.png'):
            namelist_obj_depth.append(filename)

        # for filename in Path(os.path.join('../Object-Renderer-pyrender/examples/images/bottle')).rglob(
        #         'color/*.png'):
        #     namelist_obj_color.append(filename)
        # for filename in Path(os.path.join('../Object-Renderer-pyrender/examples/images/bottle')).rglob(
        #         'depth/*.png'):
        #     namelist_obj_depth.append(filename)

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

        pair_A_RGBD = np.zeros((len_dataset, 256, 256, 4))
        pair_B_DM = np.zeros((len_dataset, 256, 256, 2))

        save_cnt = 0
        itr = 0

        p_total = np.random.permutation(len_dataset)
        test_len = int(len_dataset / 10)
        train_len = len_dataset - test_len

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


            idx_obj = np.random.randint(0, len_obj)
            idx_tex = np.random.randint(0, len_texture)
            obj_c = np.array(Image.open(str(namelist_obj_color[idx_obj])))
            obj_d = np.array(Image.open(str(namelist_obj_depth[idx_obj])))
            tex = np.array(Image.open(str(texture_list[idx_tex])))
            tex = cv2.resize(tex, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

            hand_c = hand_c[:, :, :-1]
            hand_d = hand_d[:, :, 1]
            hand_d[hand_d==1] = 0


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

            # cv2.imshow("hand_c", hand_c)
            # cv2.imshow("hand_d", np.uint8(hand_d))
            # cv2.imshow("obj_c", obj_c)
            # cv2.imshow("obj_d", np.uint8(obj_d))
            # cv2.imshow("both_c", both_c)
            # cv2.imshow("both_d", both_d)
            # cv2.imshow("mask_obj", mask_obj)
            # cv2.waitKey(0)

            masked_c = both_c.copy()
            masked_c[mask_obj == 0] = 0
            # cv2.imshow("masked_c", masked_c)
            # cv2.waitKey(0)
            """
            save as image
            """
            if i < train_len:
                name_base = "./datasets/self_2020/trial_0/wOBJ/train/"
            else:
                name_base = "./datasets/self_2020/trial_0/wOBJ/test/"

            name_Ac = name_base + "A_color/" + str(save_cnt) + ".png"
            cv2.imwrite(name_Ac, both_c)
            name_Ad = name_base + "A_depth/" + str(save_cnt) + ".png"
            cv2.imwrite(name_Ad, both_d)
            name_Bd = name_base + "B_depth/" + str(save_cnt) + ".png"
            cv2.imwrite(name_Bd, hand_d)
            name_Bm = name_base + "B_mask/" + str(save_cnt) + ".png"
            cv2.imwrite(name_Bm, mask_obj)

            name_Bmc = name_base + "A_masked_color/" + str(save_cnt) + ".png"
            cv2.imwrite(name_Bmc, masked_c)

            # A = np.concatenate((both_c, np.expand_dims(both_d, axis=-1)), axis=-1)
            # B = np.concatenate((np.expand_dims(hand_d, axis=-1), np.expand_dims(mask_obj, axis=-1)), axis=-1)
            #
            # pair_A_RGBD[save_cnt] = A
            # pair_B_DM[save_cnt] = B

            save_cnt += 1
            key = cv2.waitKey(1)

            print("itr in total --- ", (i, len_dataset, p_total[i]))

            if key == ord('q') or key == 27:
                break

    if mode == 2:
        print("mode : ", mode)
        namelist_color = []
        namelist_depth = []

        # for filename in Path(py.join('datasets/2020_seq')).rglob('c2d_ego/*.png'):
        #     namelist_color.append(filename)
        # for filename in Path(py.join('datasets/2020_seq')).rglob('depth_ego/*.png'):
        #     namelist_depth.append(filename)

        for filename in Path(py.join('datasets/2020_seq')).rglob('c2d/*.png'):
            namelist_color.append(filename)
        for filename in Path(py.join('datasets/2020_seq')).rglob('depth/*.png'):
            namelist_depth.append(filename)

        for filename in Path(py.join('datasets/2020_seq')).rglob('c2d_1/*.png'):
            namelist_color.append(filename)
        for filename in Path(py.join('datasets/2020_seq')).rglob('depth_1/*.png'):
            namelist_depth.append(filename)

        tmp = len(namelist_depth)
        len_dataset = len(namelist_depth) * 2

        d_range = [-40, 40]

        # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        # cv2.setMouseCallback('Image', click)

        # load texture data
        texture_list = []
        for filename in Path(py.join('texture')).rglob('*.png'):
            texture_list.append(filename)
        for filename in Path(py.join('texture')).rglob('*.jpg'):
            texture_list.append(filename)
        len_texture = len(texture_list)

        # main process
        key_flag = True

        pair_A_RGBD = np.zeros((len_dataset, 256, 256, 4))
        pair_B_DM = np.zeros((len_dataset, 256, 256, 2))

        save_cnt = 0
        itr = 0

        width = 640
        height = 480

        for i in range(len_dataset):
            print("itr in total --- ", (i, len_dataset))
            if i == len_dataset - 1:
                print("saving...")
                name_1 = 'pair_A_RGBD_430_1.npy'
                name_2 = 'pair_B_DM_430_1.npy'

                np.save(name_1, pair_A_RGBD)
                np.save(name_2, pair_B_DM)
                print("done")

            rgb_B = np.array(Image.open(str(namelist_color[i % tmp])))
            depth_B = np.array(Image.open(str(namelist_depth[i % tmp])))

            #test_rgb_1 = np.array(Image.open("../HPF_handtracker/dataset/EgoDexter/Desk/color/image_00000_color.png"))
            #test_rgb_2 = np.array(Image.open("../HPF_handtracker/dataset/EgoDexter/Desk/color/image_00009_color.png"))

            # preprocessing (normalizing rgb)
            rgb_B = change_rgb_order(rgb_B)




            min_d = np.min(np.nonzero(depth_B))

            if i % tmp < 39:
                d_thresh = 360
                # print("minimum d : ", min_d)
                depth_B[depth_B > min_d + d_thresh] = 0

                offset = int((width - height) / 2)
                depth_B = depth_B[:, :-2*offset]
                rgb_B = rgb_B[:, :-2*offset, :]

            else:
                d_thresh = 450
                # print("minimum d : ", min_d)
                depth_B[depth_B > min_d + d_thresh] = 0

                offset = int((width - height) / 2)
                depth_B = depth_B[:, offset:-offset]
                rgb_B = rgb_B[:, offset:-offset, :]
            # 480 480

            # normalize depth
            d_norm = 255. * (depth_B - depth_B.min()) / (depth_B.max() - depth_B.min())
            cv2.imshow("rgb", rgb_B)
            cv2.imshow("depth", np.uint8(d_norm))
            cv2.waitKey(0)

            rgb_np = cv2.resize(rgb_B, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            depth_np = cv2.resize(d_norm, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

            rgb_np[depth_np == 0] = 0

            depth_np = depth_np.astype(np.float64)
            depth_np = np.expand_dims(depth_np, axis=-1)
            min_depth = np.min(depth_np[np.nonzero(depth_np)])

            sh = np.shape(depth_np)
            depth_vis = np.zeros((256, 256, 3))
            depth_vis[:, :, 0] = np.squeeze(depth_np)

            obj_loc = int(min_depth) + (d_range[1] - d_range[0]) * np.random.random_sample() + d_range[0]

            # while key_flag:
            #     img = rgb_np
            #     cv2.imshow('Image', img)
            #     key = cv2.waitKey(1) & 0xFF
            #     if key == ord('a'):
            #         key_flag = False

            mid_x = random.uniform(80, 170)
            mid_y = random.uniform(80, 170)

            mid = (mid_x, mid_y)  # (target[1], target[0])
            # print("mid : ", mid)

            xy_list = generate_random(mid)
            contours = np.asarray(xy_list)

            result = cv2.drawContours(depth_vis, [contours], -1, (0, obj_loc, 0), cv2.FILLED)

            # randomly distribute depth value
            ratio = random.uniform(2, 6)

            for m in range(256):
                for j in range(256):
                    if result[m, j, 1] == obj_loc:
                        dif_y = abs(m - mid[0]) * 2
                        dif_x = abs(j - mid[1]) * 2
                        dist = np.sqrt(np.power(dif_y, 2) + np.power(dif_x, 2))
                        # print("dif value : ", dist / ratio)
                        result[m, j, 1] += dist / ratio

            # if original hand is closer than object, ignore that object pixel
            original_depth = np.copy(np.squeeze(depth_np))
            original_depth[original_depth == 0] = 1000
            obj_depth = result[:, :, 1]
            obj_depth[original_depth < obj_depth] = 0

            # with random texture image, overlap synthetic part
            texture = np.squeeze(np.array(Image.open(str(np.random.choice(texture_list)))))
            texture = change_rgb_order(texture)
            texture = cv2.resize(texture, (256, 256), interpolation=cv2.INTER_NEAREST)
            # cv2.imshow("target texture", texture)
            # print("texture shape ; ", np.shape(texture))
            rgb_synthetic = np.copy(rgb_np)
            depth_synthetic = np.copy(np.squeeze(depth_np))
            depth_synthetic[obj_depth > 0] = obj_depth[obj_depth > 0]

            order = np.random.randint(2)

            for a in range(sh[0]):
                for b in range(sh[1]):
                    if obj_depth[a, b]:
                        rgb_synthetic[a, b, :] = texture[a, b, :3]

            obj_depth[obj_depth > 0] = 1

            # # normalize depth to 0 ~ 1
            # minval = np.min(depth_synthetic[np.nonzero(depth_synthetic)])
            # norm_B = np.copy(depth_synthetic)
            # norm_B[np.nonzero(depth_synthetic)] -= minval
            # maxval = np.max(norm_B[norm_B < 600])
            # norm_B[norm_B >= 600] = 0
            # depth_synthetic = norm_B * 255. / float(maxval)  # 0~ 255
            #
            # minval = np.min(depth_np[np.nonzero(depth_np)])
            # norm_B = np.copy(depth_np)
            # norm_B[np.nonzero(depth_np)] -= minval
            # maxval = np.max(norm_B[norm_B < 600])
            # norm_B[norm_B >= 600] = 0
            # depth_np = norm_B * 255. / float(maxval)  # 0~ 255

            # print("minmax : ", np.min(depth_synthetic), np.max(depth_synthetic))

            # cv2.imshow("after rgb_synthetic", rgb_synthetic)
            # cv2.imshow("after depth_synthetic", np.array(depth_synthetic, dtype='u1'))
            # 
            # cv2.imshow("after rgb_np", rgb_np)
            # cv2.imshow("after depth_np", np.array(depth_np, dtype='u1'))
            # cv2.imshow("after obj_mask", obj_depth)

            depth_np = np.squeeze(depth_np.astype(np.float64))
            depth_synthetic = np.squeeze(depth_synthetic.astype(np.float64))

            # diff = np.abs(depth_np - depth_synthetic)
            # cv2.imshow("diff", np.array(diff, dtype='u1'))

            # cv2.waitKey(0)
            # print("min ", np.min(depth_B[np.nonzero(depth_B)]))
            #
            rgb_A = np.squeeze(rgb_synthetic.astype(np.float64))
            rgb_A = plt.colors.rgb_to_hsv(rgb_A)
            rgb_A[:, :, 2] = 0.5
            rgb_A = plt.colors.hsv_to_rgb(rgb_A)
            #cv2.imshow("fin ", rgb_A)

            depth_A = np.squeeze(depth_synthetic.astype(np.float64))
            depth_B_ = np.squeeze(depth_np.astype(np.float64))

            rgb_A = rgb_A * 2. - 1.0
            depth_A = depth_A * 2. / 255. - 1.0
            depth_B_ = depth_B_ * 2. / 255. - 1.0

            #print("min max ", np.min(rgb_A), np.max(rgb_A), np.min(depth_A), np.max(depth_A), np.min(depth_B_), np.max(depth_B_),np.min(obj_depth), np.max(obj_depth))

            A = np.concatenate((rgb_A, np.expand_dims(depth_A, axis=-1)), axis=-1)
            B = np.concatenate((np.expand_dims(depth_B_, axis=-1), np.expand_dims(obj_depth, axis=-1)), axis=-1)

            flip_h = np.random.randint(2)
            rotate = np.random.randint(3)

            if flip_h:
                A = np.flip(A, 1)
                B = np.flip(B, 1)

            if rotate == 0:
                A = np.rot90(A, k=1)
                B = np.rot90(B, k=1)
            elif rotate == 2:
                A = np.rot90(A, k=-1)
                B = np.rot90(B, k=-1)

            pair_A_RGBD[save_cnt] = A
            pair_B_DM[save_cnt] = B

            save_cnt += 1
            key = cv2.waitKey(1)

            if key == ord('q') or key == 27:
                break

    if mode == 1:
        namelist_color = []
        namelist_depth = []

        print("")

        for filename in Path(py.join('datasets/2020_seq')).rglob('c2d/*.png'):
            namelist_color.append(filename)
        for filename in Path(py.join('datasets/2020_seq')).rglob('depth/*.png'):
            namelist_depth.append(filename)

        for filename in Path(py.join('datasets/2020_seq')).rglob('c2d_1/*.png'):
            namelist_color.append(filename)
        for filename in Path(py.join('datasets/2020_seq')).rglob('depth_1/*.png'):
            namelist_depth.append(filename)


        tmp = len(namelist_depth)
        len_dataset = len(namelist_depth) * 4

        d_range = [-30, 35]

        #cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        #cv2.setMouseCallback('Image', click)

        # load texture data
        texture_list = []
        for filename in Path(py.join('texture')).rglob('*.png'):
            texture_list.append(filename)
        for filename in Path(py.join('texture')).rglob('*.jpg'):
            texture_list.append(filename)
        len_texture = len(texture_list)


        # main process
        key_flag = True

        pair_A_RGBD = np.zeros((len_dataset, 256, 256, 4))
        pair_B_D = np.zeros((len_dataset, 256, 256, 2))

        save_cnt = 0
        itr = 0

        width = 640
        height = 480

        for i in range(len_dataset):
            print("itr in total --- ", (i, len_dataset))
            if i == len_dataset - 1:
                print("saving...")
                name_1 = 'pair_A_RGBD_new.npy'
                name_2 = 'pair_B_DM_new.npy'

                np.save(name_1, pair_A_RGBD)
                np.save(name_2, pair_B_D)
                print("done")

            rgb_B = np.array(Image.open(str(namelist_color[i%tmp])))
            depth_B = np.array(Image.open(str(namelist_depth[i%tmp])))

            min_d = np.min(np.nonzero(depth_B))
            d_thresh = 450
            #print("minimum d : ", min_d)
            depth_B[depth_B > min_d + d_thresh] = 0

            offset = int((width - height) / 2)
            depth_B = depth_B[:, offset:-offset]
            rgb_B = rgb_B[:, offset:-offset, :]
            # 480 480

            # loading with pillow change channel order
            B_rgb_temp = np.copy(rgb_B)
            B_rgb_temp[:, :, 0] = rgb_B[:, :, 2]
            B_rgb_temp[:, :, 2] = rgb_B[:, :, 0]
            rgb_B = B_rgb_temp

            # normalize depth
            d_norm = 255. * (depth_B - depth_B.min()) / (depth_B.max() - depth_B.min())

            rgb_np = cv2.resize(rgb_B, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            depth_np = cv2.resize(d_norm, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

            rgb_np[depth_np == 0] = 0

            depth_np = depth_np.astype(np.float64)
            depth_np = np.expand_dims(depth_np, axis=-1)
            min_depth = np.min(depth_np[np.nonzero(depth_np)])

            sh = np.shape(depth_np)
            depth_vis = np.zeros((256, 256, 3))
            depth_vis[:, :, 0] = np.squeeze(depth_np)

            obj_loc = int(min_depth) + (d_range[1] - d_range[0]) * np.random.random_sample() + d_range[0]

            # while key_flag:
            #     img = rgb_np
            #     cv2.imshow('Image', img)
            #     key = cv2.waitKey(1) & 0xFF
            #     if key == ord('a'):
            #         key_flag = False

            mid_x = random.uniform(55, 160)
            mid_y = random.uniform(55, 160)

            mid = (mid_x, mid_y) #(target[1], target[0])
            #print("mid : ", mid)

            xy_list = generate_random(mid)
            contours = np.asarray(xy_list)

            result = cv2.drawContours(depth_vis, [contours], -1, (0, obj_loc, 0), cv2.FILLED)

            # randomly distribute depth value
            ratio = random.uniform(2, 6)

            for m in range(256):
                for j in range(256):
                    if result[m, j, 1] == obj_loc:
                        dif_y = abs(m - mid[0]) * 2
                        dif_x = abs(j - mid[1]) * 2
                        dist = np.sqrt(np.power(dif_y, 2) + np.power(dif_x, 2))
                        #print("dif value : ", dist / ratio)
                        result[m, j, 1] += dist / ratio


            # if original hand is closer than object, ignore that object pixel
            original_depth = np.copy(np.squeeze(depth_np))
            original_depth[original_depth == 0] = 1000
            obj_depth = result[:, :, 1]
            obj_depth[original_depth < obj_depth] = 0

            # with random texture image, overlap synthetic part
            texture = np.squeeze(np.array(Image.open(str(np.random.choice(texture_list)))))
            texture = cv2.resize(texture, (256, 256), interpolation=cv2.INTER_NEAREST)
            #cv2.imshow("target texture", texture)
            #print("texture shape ; ", np.shape(texture))
            rgb_synthetic = np.copy(rgb_np)
            depth_synthetic = np.copy(np.squeeze(depth_np))
            depth_synthetic[obj_depth > 0] = obj_depth[obj_depth > 0]

            order = np.random.randint(2)

            for a in range(sh[0]):
                for b in range(sh[1]):
                    if obj_depth[a, b]:
                        if order:
                            rgb_synthetic[a, b, 0] = texture[a, b, 2]
                            rgb_synthetic[a, b, 1] = texture[a, b, 1]
                            rgb_synthetic[a, b, 2] = texture[a, b, 0]
                        else:
                            rgb_synthetic[a, b, 0] = texture[a, b, 0]
                            rgb_synthetic[a, b, 1] = texture[a, b, 1]
                            rgb_synthetic[a, b, 2] = texture[a, b, 2]

            obj_depth[obj_depth > 0] = 1

            # # normalize depth to 0 ~ 1
            # minval = np.min(depth_synthetic[np.nonzero(depth_synthetic)])
            # norm_B = np.copy(depth_synthetic)
            # norm_B[np.nonzero(depth_synthetic)] -= minval
            # maxval = np.max(norm_B[norm_B < 600])
            # norm_B[norm_B >= 600] = 0
            # depth_synthetic = norm_B * 255. / float(maxval)  # 0~ 255
            #
            # minval = np.min(depth_np[np.nonzero(depth_np)])
            # norm_B = np.copy(depth_np)
            # norm_B[np.nonzero(depth_np)] -= minval
            # maxval = np.max(norm_B[norm_B < 600])
            # norm_B[norm_B >= 600] = 0
            # depth_np = norm_B * 255. / float(maxval)  # 0~ 255

            #print("minmax : ", np.min(depth_synthetic), np.max(depth_synthetic))

            # cv2.imshow("after rgb_synthetic", rgb_synthetic)
            # cv2.imshow("after depth_synthetic", np.array(depth_synthetic, dtype='u1'))
            #
            # cv2.imshow("after rgb_np", rgb_np)
            # cv2.imshow("after depth_np", np.array(depth_np, dtype='u1'))
            # cv2.imshow("after obj_mask", obj_depth)

            depth_np = np.squeeze(depth_np.astype(np.float64))
            depth_synthetic = np.squeeze(depth_synthetic.astype(np.float64))

            #diff = np.abs(depth_np - depth_synthetic)
            #cv2.imshow("diff", np.array(diff, dtype='u1'))

            #cv2.waitKey(0)
            #print("min ", np.min(depth_B[np.nonzero(depth_B)]))
            #
            rgb_A = np.squeeze(rgb_synthetic.astype(np.float64))
            depth_A = np.squeeze(depth_synthetic.astype(np.float64))

            depth_B_ = np.squeeze(depth_np.astype(np.float64))

            rgb_A = rgb_A * 2. / 255. - 1.0
            depth_A = depth_A * 2. / 255. - 1.0
            depth_B_ = depth_B_ * 2. / 255. - 1.0

            A = np.concatenate((rgb_A, np.expand_dims(depth_A, axis=-1)), axis=-1)
            B = np.concatenate((np.expand_dims(depth_B_, axis=-1), np.expand_dims(obj_depth, axis=-1)), axis=-1)

            flip_h = np.random.randint(2)
            rotate = np.random.randint(3)

            if flip_h:
                A = np.flip(A, 1)
                B = np.flip(B, 1)

            if rotate == 0:
                A = np.rot90(A, k=1)
                B = np.rot90(B, k=1)
            elif rotate == 2:
                A = np.rot90(A, k=-1)
                B = np.rot90(B, k=-1)

            pair_A_RGBD[save_cnt] = A
            pair_B_D[save_cnt] = B

            save_cnt += 1
            key = cv2.waitKey(1)

            if key == ord('q') or key == 27:
                break

    if mode == 0:
        pair_B_namelist_color = []
        pair_B_namelist_depth = []

        for filename in Path(py.join('datasets/hand/train_pair_withoutObj_B')).rglob('c2d/*.png'):
            pair_B_namelist_color.append(filename)
        for filename in Path(py.join('datasets/hand/train_pair_withoutObj_B')).rglob('depth/*.png'):
            pair_B_namelist_depth.append(filename)

        tmp = len(pair_B_namelist_depth)
        len_dataset = len(pair_B_namelist_depth) * 2

        d_range = [-10, 40]

        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Image', click)

        # load texture data
        texture_list = []
        for filename in Path(py.join('texture')).rglob('*.png'):
            texture_list.append(filename)
        for filename in Path(py.join('texture')).rglob('*.jpg'):
            texture_list.append(filename)
        len_texture = len(texture_list)


        # main process
        key_flag = True

        pair_A_RGBD = np.zeros((len_dataset, 240, 240, 4))
        pair_B_D = np.zeros((len_dataset, 240, 240, 2))
        #pair_B_RGBDM = np.zeros((len_dataset, 240, 240, 5))

        save_cnt = 0
        itr = 0

        for i in range(len_dataset):
            print("i = ", i)
            if i == len_dataset - 1:
                name_1 = 'pair_A_RGBD_1.npy'
                name_2 = 'pair_B_DM_1.npy'

                np.save(name_1, pair_A_RGBD)
                np.save(name_2, pair_B_D)

            rgb_B = np.array(Image.open(str(pair_B_namelist_color[i%tmp])))
            depth_B = np.array(Image.open(str(pair_B_namelist_depth[i%tmp])))

            depth_B[depth_B > 700] = 0

            offset = int((320 - 240) / 2)
            depth_B = depth_B[:, offset:-offset]
            rgb_B = rgb_B[:, offset:-offset, :]

            B_rgb_temp = np.copy(rgb_B)
            B_rgb_temp[:, :, 0] = rgb_B[:, :, 2]
            B_rgb_temp[:, :, 2] = rgb_B[:, :, 0]
            rgb_B = B_rgb_temp

            depth_B = np.expand_dims(depth_B, axis=-1)
            min_depth = np.min(depth_B[np.nonzero(depth_B)])

            sh = np.shape(depth_B)
            depth_vis = np.zeros((240, 240, 3))
            depth_vis[:, :, 0] = np.squeeze(depth_B)

            obj_loc = min_depth + (d_range[1] - d_range[0]) * np.random.random_sample() + d_range[0]
            """
            while key_flag:
                img = rgb_B
                cv2.imshow('Image', img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('a'):
                    key_flag = False
            """
            mid_x = random.uniform(65, 160)
            mid_y = random.uniform(75, 170)

            mid = (mid_x, mid_y) #(target[1], target[0])
            #print("mid : ", mid)

            xy_list = generate_random(mid)
            contours = np.asarray(xy_list)

            result = cv2.drawContours(depth_vis, [contours], -1, (0, obj_loc, 0), cv2.FILLED)

            # randomly distribute depth value
            ratio = random.uniform(2, 6)

            for m in range(240):
                for j in range(240):
                    if result[m, j, 1] == obj_loc:
                        dif_y = abs(m - mid[0]) * 2
                        dif_x = abs(j - mid[1]) * 2
                        dist = np.sqrt(np.power(dif_y, 2) + np.power(dif_x, 2))
                        #print("dif value : ", dist / ratio)
                        result[m, j, 1] += dist / ratio


            # if original hand is closer than object, ignore that object pixel
            original_depth = np.copy(np.squeeze(depth_B))
            original_depth[original_depth == 0] = 1000
            obj_depth = result[:, :, 1]
            obj_depth[original_depth < obj_depth] = 0

            # with random texture image, overlap synthetic part
            texture = np.squeeze(np.array(Image.open(str(np.random.choice(texture_list)))))
            texture = cv2.resize(texture, (240, 240), interpolation=cv2.INTER_AREA)
            #cv2.imshow("target texture", texture)
            #print("texture shape ; ", np.shape(texture))
            rgb_synthetic = np.copy(rgb_B)
            depth_synthetic = np.copy(np.squeeze(depth_B))
            depth_synthetic[obj_depth > 0] = obj_depth[obj_depth > 0]

            order = np.random.randint(2)

            for a in range(sh[0]):
                for b in range(sh[1]):
                    if obj_depth[a, b]:
                        if order:
                            rgb_synthetic[a, b, 0] = texture[a, b, 2]
                            rgb_synthetic[a, b, 1] = texture[a, b, 1]
                            rgb_synthetic[a, b, 2] = texture[a, b, 0]
                        else:
                            rgb_synthetic[a, b, 0] = texture[a, b, 0]
                            rgb_synthetic[a, b, 1] = texture[a, b, 1]
                            rgb_synthetic[a, b, 2] = texture[a, b, 2]

            obj_depth[obj_depth > 0] = 1

            # normalize depth to 0 ~ 1
            minval = np.min(depth_synthetic[np.nonzero(depth_synthetic)])
            norm_B = np.copy(depth_synthetic)
            norm_B[np.nonzero(depth_synthetic)] -= minval
            maxval = np.max(norm_B[norm_B < 600])
            norm_B[norm_B >= 600] = 0
            depth_synthetic = norm_B * 255. / float(maxval)  # 0~ 255

            minval = np.min(depth_B[np.nonzero(depth_B)])
            norm_B = np.copy(depth_B)
            norm_B[np.nonzero(depth_B)] -= minval
            maxval = np.max(norm_B[norm_B < 600])
            norm_B[norm_B >= 600] = 0
            depth_B = norm_B * 255. / float(maxval)  # 0~ 255
            #
            # cv2.imshow("after rgb_synthetic", rgb_synthetic)
            # cv2.imshow("after depth_synthetic", np.array(depth_synthetic, dtype='u1'))
            #
            # cv2.imshow("after rgb_B", rgb_B)
            # cv2.imshow("after depth_B", np.array(depth_B, dtype='u1'))
            # cv2.imshow("after obj_mask", obj_depth)
            #
            # cv2.waitKey(0)
            #print("min ", np.min(depth_B[np.nonzero(depth_B)]))

            rgb_synthetic = rgb_synthetic * 2. / 255. - 1.0
            rgb_B = rgb_B * 2. / 255. - 1.0

            depth_synthetic = depth_synthetic / 255.
            depth_B = depth_B / 255.

            A = np.concatenate((rgb_synthetic, np.expand_dims(depth_synthetic, axis=-1)), axis=-1)
            B = np.concatenate((depth_B, np.expand_dims(obj_depth, axis=-1)), axis=-1)
            #B = np.copy(depth_B)

            flip_h = np.random.randint(2)
            rotate = np.random.randint(3)

            if flip_h:
                A = np.flip(A, 1)
                B = np.flip(B, 1)

            if rotate == 0:
                A = np.rot90(A, k=1)
                B = np.rot90(B, k=1)
            elif rotate == 2:
                A = np.rot90(A, k=-1)
                B = np.rot90(B, k=-1)

            pair_A_RGBD[save_cnt] = A
            pair_B_D[save_cnt] = B

            save_cnt += 1
            key = cv2.waitKey(1)

            if key == ord('q') or key == 27:
                break
