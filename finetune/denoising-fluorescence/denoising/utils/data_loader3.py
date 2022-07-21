import os
import numpy as np
from PIL import Image
import numbers
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor, _is_pil_image
from torchvision.datasets.folder import has_file_allowed_extension
import sys
import json
from tifffile import imread
from pprint import pprint
from time import time
import random
import cv2



class CareLoader(torch.utils.data.Dataset):

    def __init__(self, root, train='train',
                 transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.train = train
        self.img_list, self.gt_list = self._gather_files()
        print(f'train file:{len(self.img_list)}')
        self.transform = transform
        self.target_transform = target_transform

    def _gather_files(self):
        file = np.load(os.path.join(self.root, 'data_label.npz'))
        img_list = file['X']
        gt_list = file['Y']
        length = len(img_list)
        if self.train == 'train':

            img_list = img_list[:int(length * 0.8)]
            gt_list = gt_list[:int(length * 0.8)]
        elif self.train == 'val':
            img_list = img_list[int(length * 0.8):int(length * 0.9)]
            gt_list = gt_list[int(length * 0.8):int(length * 0.9) ]
        else:
            img_list = img_list[int(length * 0.9):]
            gt_list = gt_list[int(length * 0.9):]
        print(f'img list length{len(img_list)}, gt list length{len(gt_list)}')
        return img_list, gt_list

    def __len__(self):
        return len(self.img_list)

    def randomCrop(self, img, gt, size, scale):
        assert img.shape[0] >= size
        assert img.shape[1] >= size
        # print(img.shape, gt.shape)
        x = np.random.randint(0, img.shape[1] - size + 1)
        y = np.random.randint(0, img.shape[0] - size + 1)
        imgOut = img[y:y + size, x:x + size].copy()
        imgOutC = gt[y:y + size * scale, x:x + size * scale].copy()
        return imgOut, imgOutC

    def norm(self, img, percentage_low, percentage_high):
        img = (img - np.percentile(img, percentage_low)) / (
                np.percentile(img, percentage_high) - np.percentile(img, percentage_low) + 1e-7)
        img[img > 1] = 1
        img[img < 0] = 0
        return img

    def aug(self, img, gt):
        # img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        # gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_LINEAR)
        if self.train == 'train':
            # img, gt = self.randomCrop(img, gt, size=128, scale=1)
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot90 = random.random() < 0.5
            if hflip:
                img = np.flip(img, 1)
                gt = np.flip(gt, 1)
            if vflip:
                img = np.flip(img, 0)
                gt = np.flip(gt, 0)
            if rot90:
                img = np.rot90(img)
                gt = np.rot90(gt)
        img = self.norm(img, percentage_low=2, percentage_high=99)
        gt = self.norm(gt, percentage_low=2, percentage_high=99)
        img = torch.tensor(img, dtype=torch.float).unsqueeze_(dim=0)
        gt = torch.tensor(gt, dtype=torch.float).unsqueeze_(dim=0)
        return img, gt

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        noisy, clean = self.img_list[index], self.gt_list[index]
        #noisy = imread(noisy)
        #clean = imread(clean)
        noisy = np.squeeze(noisy, axis=0)
        clean = np.squeeze(clean, axis=0)
        noisy, clean = self.aug(noisy, clean)
        return noisy, clean


def get_train_loader(data_root, batch_size, test_batch_size):
    train_dataset = CareLoader(data_root, train='train')
    val_datasdet = CareLoader(data_root, train='val')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_datasdet, batch_size=test_batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

def get_test_loader(data_root):
    test_loader = CareLoader(data_root, train='test')
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=1, shuffle=False, num_workers=1)
    return test_loader