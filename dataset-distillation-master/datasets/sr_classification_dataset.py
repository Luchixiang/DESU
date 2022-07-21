from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tifffile import imread
import numpy as np
import cv2
import random
import torch


class DESUDataset(Dataset):
    def __init__(self, img_list, train=False):
        self.train = train

        sample_img = img_list[0]

        if sample_img.endswith('.tif'):
            img = Image.fromarray(imread(sample_img).astype(np.float))
        elif sample_img.endswith('.png'):
            img = Image.open(sample_img)
        width, height = img.size
        img_list = self.generate_crop(img_list, width, height, patch_size=256, stride=256)
        self.img_list, self.gt_list = self.generate_rotation(img_list)
        ids = np.random.permutation(self.img_list.shape[0])
        self.img_list_pemuted = self.img_list[ids]
        self.gt_list_pemuted = self.gt_list[ids]
        # self.resize_transforms = [transforms.Compose([transforms.Resize(224, 224)])]
        self.img_max, self.img_min = self.get_max_min(self.img_list)

    def __len__(self):
        return self.img_list.shape[0]

    def generate_rotation(self, img_list):
        rotated_list = []
        gt_list = []
        for img in img_list:
            i = np.random.randint(0, 4)
            rotated_list.append(np.rot90(img, k=i))
            gt_list.append(i)
        return np.stack(rotated_list), np.stack(gt_list)

    def generate_crop(self, img_list, width, height, patch_size=256, stride=128):
        # width, height = img.size
        print(f'sample image size{width}, {height}')
        if width > patch_size:
            print('generating the crop')
            cropped_list = []
            num_patches = 1
            for img in img_list:
                if img.endswith('.tif'):
                    img = imread(img).astype(np.float)

                elif img.endswith('.png'):
                    img = np.array(Image.open(img))
                    (h, w) = np.shape(img)
                h, w = img.shape
                idx_x = 0
                while (idx_x + patch_size < h):
                    idx_y = 0
                    while (idx_y + patch_size < w):
                        patch = img[idx_x:idx_x + patch_size, idx_y:idx_y + patch_size]
                        cropped_list.append(patch)
                        num_patches = num_patches + 1
                        idx_y = idx_y + stride
                    idx_x = idx_x + stride
            return cropped_list
        else:
            np_list = []
            for img in img_list:
                if img.endswith('.tif'):
                    img = imread(img).astype(np.float)
                elif img.endswith('.png'):
                    img = np.array(Image.open(img))
                    (h, w) = np.shape(img)
                np_list.append(img)
            return np_list

    def get_max_min(self, img_list):
        array = np.array(img_list)
        print('array shape:', array.shape)
        min_intensity = np.percentile(array, 2)
        max_intensity = np.percentile(array, 98)
        print(f'max intensity{max_intensity}, min_intensity{min_intensity}')
        return max_intensity, min_intensity

    def __getitem__(self, item):
        img = self.img_list_pemuted[item]
        gt = self.gt_list_pemuted[item]
        if isinstance(img, str):
            if img.endswith('.tif'):
                img = imread(img).astype(np.float)
            elif img.endswith('.png'):
                img = np.array(Image.open(img))
            else:
                print(img)
                raise Exception('file format not supported')
        img = self.aug(img)
        return torch.from_numpy(img).unsqueeze_(dim=0).float(), torch.tensor(gt).long()

    # def norm(self, img, percentage_low, percentage_high):
    #     img = (img - np.percentile(img, percentage_low)) / (
    #             np.percentile(img, percentage_high) - np.percentile(img, percentage_low) + 1e-7)
    #     img[img > 1] = 1
    #     img[img < 0] = 0
    #     return img

    def aug(self, img):
        img = cv2.resize(img, (224, 224))
        img = (img - self.img_min) / (self.img_max - self.img_min + 1e-7)
        img = img.clip(0, 1)
        # img = self.norm(img, percentage_low=2, percentage_high=99)
        # gt = self.norm(gt, percentage_low=2, percentage_high=99)
        return img


class TissueDataset(Dataset):
    def __init__(self, img_list, gt_list, transform=None):
        # self.img_list = img_list
        # self.gt_list = gt_list
        # self.transform = transform
        sample_img = img_list[0]
        if sample_img.endswith('.tif'):
            img = Image.fromarray(imread(sample_img).astype(np.float))
        elif sample_img.endswith('.png'):
            img = Image.open(sample_img)
        width, height = img.size
        self.img_list, self.gt_list= self.generate_crop(img_list, gt_list, width, height, patch_size=256, stride=256)
        self.img_max, self.img_min = self.get_max_min(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def get_max_min(self, img_list):
        array = np.array(img_list)
        print('array shape:', array.shape)
        min_intensity = np.percentile(array, 2)
        max_intensity = np.percentile(array, 98)
        print(f'max intensity{max_intensity}, min_intensity{min_intensity}')
        return max_intensity, min_intensity

    def generate_crop(self, img_list,gt_list, width, height, patch_size=256, stride=128):
        # width, height = img.size
        print(f'sample image size{width}, {height}')
        if width > patch_size:
            print('generating the crop')
            cropped_list = []
            cropped_gt_list = []
            num_patches = 1
            for i in range(len(img_list)):
                img = img_list[i]
                if img.endswith('.tif'):
                    img = imread(img).astype(np.float)

                elif img.endswith('.png'):
                    img = np.array(Image.open(img))
                    (h, w) = np.shape(img)
                h, w = img.shape
                idx_x = 0
                while (idx_x + patch_size < h):
                    idx_y = 0
                    while (idx_y + patch_size < w):
                        patch = img[idx_x:idx_x + patch_size, idx_y:idx_y + patch_size]
                        cropped_list.append(patch)
                        num_patches = num_patches + 1
                        idx_y = idx_y + stride
                        cropped_gt_list.append(gt_list[i])
                    idx_x = idx_x + stride
            return cropped_list, cropped_gt_list
        else:
            np_list = []
            for img in img_list:
                if img.endswith('.tif'):
                    img = imread(img).astype(np.float)
                elif img.endswith('.png'):
                    img = np.array(Image.open(img))
                    (h, w) = np.shape(img)
                np_list.append(img)
            return np_list, gt_list

    def __getitem__(self, item):
        img = self.img_list[item]
        gt = self.gt_list[item]
        if isinstance(img, str):
            if img.endswith('.tif'):
                img = imread(img).astype(np.float)
            elif img.endswith('.png'):
                img = np.array(Image.open(img))
            else:
                print(img)
                raise Exception('file format not supported')
        img = self.aug(img)
        return torch.from_numpy(img).unsqueeze_(dim=0).float(), torch.tensor(gt).long()

    #  def random_crop(self, ):
    def aug(self, img):
        img = cv2.resize(img, (224, 224))
        img = (img - self.img_min) / (self.img_max - self.img_min + 1e-7)
        img = img.clip(0, 1)
        # img = self.norm(img, percentage_low=2, percentage_high=99)
        # gt = self.norm(gt, percentage_low=2, percentage_high=99)
        return img
