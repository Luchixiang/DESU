import os
import os.path
import numpy as np
import h5py
import torch
import cv2
import torch.utils.data as udata
import random
from utils import augment
from tifffile import imread


class Dataset_modified(udata.Dataset):
    def __init__(self, data_path, target_path, img_avg=1, patch_size=64, stride=32, train=True):
        super(Dataset_modified, self).__init__()
        self.img_list, self.target_list = self.generate_patch(data_path, target_path, patch_size, stride, img_avg)
        self.train = train

    def __len__(self):
        return len(self.img_list)

    def augment(self, img, target):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5
        if hflip:
            img = cv2.flip(img, 1)
            target = cv2.flip(target, 1)
        if vflip:
            img = cv2.flip(img, 0)
            target = cv2.flip(target, 0)
        if rot90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            target = cv2.rotate(target, cv2.ROTATE_90_CLOCKWISE)
        return img, target

    def __getitem__(self, item):
        img = self.img_list[item]
        target = self.target_list[item]
        if self.train:
            if random.random() < 0.5:
                [img, target] = self.augment(img, target)
                img = img.copy()
                target = target.copy()
        img = self.norm(img, 1, 99)
        target = self.norm(target, 1, 99)
        img = np.expand_dims(img, axis=0)
        target = np.expand_dims(target, axis=0)
       #  print(np.max(img), np.min(img))
        return [torch.from_numpy(img).float(), torch.from_numpy(target).float()]

    def norm(self, img, percentage_low, percentage_high):
        # img = (img - np.percentile(img, percentage_low)) / (
        #         np.percentile(img, percentage_high) - np.percentile(img, percentage_low) + 1e-7)
        # img[img > 1] = 1
        # img[img < 0] = 0
        img = (img - 154.5) / 66.028
        img = img/ 255.
        return img

    def generate_patch(self, data_path, target_path, patch_size, stride, img_avg):
        print('Generating data patch')
        img_list = []
        num_patches = 0
        patch_size_target = patch_size
        stride_target = stride
        if 'hr' in data_path:
            patch_size_target = patch_size_target * 2
            stride_target = stride_target * 2
        target_list = []
        for image in os.listdir(data_path):
           # print(image, image.split('_')[1][3:-4])
            # if int(image[-5]) != img_avg and int(image[-6:-4]) != img_avg:
            #     continue
            if int(image.split('_')[1][3:-4]) != img_avg:
                continue
            img = imread(os.path.join(data_path, image))
            target = imread(os.path.join(target_path, image))
            (h, w) = np.shape(img)

            idx_x = 0
            while (idx_x + patch_size < h):
                idx_y = 0
                while (idx_y + patch_size < w):
                    patch = img[idx_x:idx_x + patch_size, idx_y:idx_y + patch_size]
                    img_list.append(patch)
                    num_patches = num_patches + 1

                    idx_y = idx_y + stride

                idx_x = idx_x + stride
            (h, w) = np.shape(target)
            idx_x = 0
            while (idx_x + patch_size_target < h):
                idx_y = 0
                while (idx_y + patch_size_target < w):
                    patch = target[idx_x:idx_x + patch_size_target, idx_y:idx_y + patch_size_target]
                    target_list.append(patch)
                    num_patches = num_patches + 1
                    idx_y = idx_y + stride_target
                idx_x = idx_x + stride_target

        return img_list, target_list


class Dataset(udata.Dataset):
    # Dataset to load pairs of noisy/target(400) images

    def __init__(self, img_avg=1, patch_size=64, stride=32):
        ''' img_avg is the number of raw images averaged to create this image
            patch_size and stride determine the patching strategy
        '''

        super(Dataset, self).__init__()

        self.noisy_file_name = f'../net_data/avg{img_avg}_{patch_size}_{stride}.h5'
        self.target_file_name = f'../net_data/avg{400}_{patch_size}_{stride}.h5'

    def __len__(self):
        h5f_noisy = h5py.File(self.noisy_file_name, 'r')
        h5f_target = h5py.File(self.target_file_name, 'r')
        if h5f_noisy['data'].shape[0] != h5f_target['data'].shape[0]:
            raise NotImplemented('The noisy and the target files have different length.')
        else:
            return h5f_noisy['data'].shape[0]

    def __getitem__(self, index):
        h5f_noisy = h5py.File(self.noisy_file_name, 'r')
        h5f_target = h5py.File(self.target_file_name, 'r')

        data_noisy = np.array(h5f_noisy['data'][index, :])
        data_target = np.array(h5f_target['data'][index, :])

        h5f_noisy.close()
        h5f_target.close()

        if random.random() < 0.5:
            [data_noisy, data_target] = augment([data_noisy, data_target], True, True)
            data_noisy = data_noisy.copy()
            data_target = data_target.copy()

        return [torch.from_numpy(data_noisy), torch.from_numpy(data_target)]
