from PIL import Image
import glob
import numpy as np
import torchvision.transforms as transforms
import cv2


def resize_and_crop_np(img, crop_h, crop_w, top, left):
    # for numpy array from tensor, C * W * H
    img = img[:, int(top):int(top) + crop_h, int(left):int(left) + crop_w]
    img = img.transpose((1, 2, 0))
    return cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)


class InpaintDataset():
    def __init__(self, phase, view):
        self.phase = phase
        self.view = view
        transform_list = [ transforms.ToTensor() ]
        self.transform = transforms.Compose(transform_list)

        # wOBJ_init : train/test - ego/exo
        # woOBJ_init : ego/exo

        dir_self = "./datasets/self_2020/wOBJ_init/" + str(phase) + "/" + str(view)
        self.files_self_A_rgb = sorted(glob.glob(dir_self + '/A_color' + '/*.*'))
        self.files_self_A_depth = sorted(glob.glob(dir_self + '/A_depth' + '/*.*'))
        self.files_self_B_depth = sorted(glob.glob(dir_self + '/B_depth' + '/*.*'))
        self.files_self_B_mask = sorted(glob.glob(dir_self + '/B_mask' + '/*.*'))
        self.files_self_A_m_rgb = sorted(glob.glob(dir_self + '/A_masked_color' + '/*.*'))


    def __getitem__(self, index):
        # self dataset, rgb is already masked by depth
        idx = index % len(self.files_self_A_rgb)

        A_rgb = np.asarray(Image.open(self.files_self_A_rgb[idx]).convert('RGB'))
        A_d = np.asarray((Image.open(self.files_self_A_depth[idx])))   # 0~255, closer = smaller
        B_d = np.asarray((Image.open(self.files_self_B_depth[idx])))
        obj_rgb = np.asarray((Image.open(self.files_self_A_m_rgb[idx]).convert('RGB')))

        seed = np.random.randint(4)
        if seed == 0:
            A_rgb = np.flip(A_rgb, (0, 1))
            A_d = np.flip(A_d, (0, 1))
            B_d = np.flip(B_d, (0, 1))
            obj_rgb = np.flip(obj_rgb, (0, 1))
        elif seed == 1:
            A_rgb = np.flip(A_rgb, 0)
            A_d = np.flip(A_d, 0)
            B_d = np.flip(B_d, 0)
            obj_rgb = np.flip(obj_rgb, 0)
        elif seed == 2:
            A_rgb = np.flip(A_rgb, 1)
            A_d = np.flip(A_d, 1)
            B_d = np.flip(B_d, 1)
            obj_rgb = np.flip(obj_rgb, 1)

        A_rgbd = np.concatenate((A_rgb, np.expand_dims(A_d, axis=-1)), axis=-1)

        A_rgbd_name = "./datasets/self_2020/wOBJ_processed/" + str(self.phase) + "/A_rgbd/" + str(self.view) + "_" + str(index) + ".png"
        obj_rgb_name = "./datasets/self_2020/wOBJ_processed/" + str(self.phase) + "/obj_rgb/" + str(self.view) + "_" + str(index) + ".png"
        B_d_name = "./datasets/self_2020/wOBJ_processed/" + str(self.phase) + "/B_d/" + str(self.view) + "_" + str(index) + ".png"

        #A_rgb = A_rgb[:, :, ::-1].copy()
        #cv2.imshow("rgb", A_rgb)
        #cv2.waitKey(0)

        cv2.imwrite(A_rgbd_name, A_rgbd)
        cv2.imwrite(obj_rgb_name, obj_rgb)
        cv2.imwrite(B_d_name, B_d)


        return None

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.files_self_A_rgb)


class InpaintDataset_woOBJ():
    def __init__(self, phase, view):
        self.phase = phase
        self.view = view
        transform_list = [ transforms.ToTensor() ]
        self.transform = transforms.Compose(transform_list)

        # wOBJ_init : train/test - ego/exo
        # woOBJ_init : ego/exo

        dir_self = "./datasets/self_2020/woOBJ_init/" + str(view)
        self.files_self_A_rgb = sorted(glob.glob(dir_self + '/A_color' + '/*.*'))
        self.files_self_A_depth = sorted(glob.glob(dir_self + '/A_depth' + '/*.*'))
        self.files_self_B_depth = sorted(glob.glob(dir_self + '/B_depth' + '/*.*'))

        self.datasize = int(len(self.files_self_A_rgb))

        self.p_total = np.random.permutation(self.datasize)
        self.test_len = int(self.datasize/10)




    def __getitem__(self, index):
        # self dataset, rgb is already masked by depth
        idx = self.p_total[index]

        A_rgb = np.asarray(Image.open(self.files_self_A_rgb[idx]).convert('RGB'))
        A_d = np.asarray((Image.open(self.files_self_A_depth[idx])))  # 0~255, closer = smaller
        B_d = np.asarray((Image.open(self.files_self_B_depth[idx])))

        obj_rgb = np.zeros(np.shape(A_rgb))

        seed = np.random.randint(4)
        if seed == 0:
            A_rgb = np.flip(A_rgb, (0, 1))
            A_d = np.flip(A_d, (0, 1))
            B_d = np.flip(B_d, (0, 1))
        elif seed == 1:
            A_rgb = np.flip(A_rgb, 0)
            A_d = np.flip(A_d, 0)
            B_d = np.flip(B_d, 0)
        elif seed == 2:
            A_rgb = np.flip(A_rgb, 1)
            A_d = np.flip(A_d, 1)
            B_d = np.flip(B_d, 1)

        A_rgbd = np.concatenate((A_rgb, np.expand_dims(A_d, axis=-1)), axis=-1)

        if index < self.test_len:
            phase = 'test'
        else:
            phase = 'train'

        A_rgbd_name = "./datasets/self_2020/woOBJ_processed/" + str(phase) + "/A_rgbd/" + str(
            self.view) + "_" + str(index) + ".png"
        obj_rgb_name = "./datasets/self_2020/woOBJ_processed/" + str(phase) + "/obj_rgb/" + str(
            self.view) + "_" + str(index) + ".png"
        B_d_name = "./datasets/self_2020/woOBJ_processed/" + str(phase) + "/B_d/" + str(self.view) + "_" + str(
            index) + ".png"

        # A_rgb = A_rgb[:, :, ::-1].copy()
        # cv2.imshow("rgb", A_rgb)
        # cv2.waitKey(0)

        cv2.imwrite(A_rgbd_name, A_rgbd)
        cv2.imwrite(obj_rgb_name, obj_rgb)
        cv2.imwrite(B_d_name, B_d)

        return None

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.files_self_A_rgb)


if __name__ == '__main__':

    dataset = InpaintDataset('test', 'exo')
    for i, data in enumerate(dataset):
        if i % 100 == 0:
            print("i test exo : ", i)
        if i == len(dataset):
            break

    dataset = InpaintDataset('test', 'ego')
    for i, data in enumerate(dataset):
        if i % 100 == 0:
            print("i test ego : ", i)
        if i == len(dataset):
            break

    dataset = InpaintDataset('train', 'exo')
    for i, data in enumerate(dataset):
        if i % 100 == 0:
            print("i train exo : ", i)
        if i == len(dataset):
            break

    dataset = InpaintDataset('train', 'ego')
    for i, data in enumerate(dataset):
        if i % 100 == 0:
            print("i train ego : ", i)
        if i == len(dataset):
            break



    """
    dataset = InpaintDataset_woOBJ('-', 'exo')
    for i, data in enumerate(dataset):
        if i % 100 == 0:
            print("i test exo : ", i)
        if i == len(dataset):
            break

    dataset = InpaintDataset_woOBJ('-', 'ego')
    for i, data in enumerate(dataset):
        if i % 100 == 0:
            print("i test ego : ", i)
        if i == len(dataset):
            break
    """