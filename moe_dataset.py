import os
from torch.utils.data import DataLoader, Dataset
from tifffile import imread
import numpy as np
from PIL import Image
import random
import torch
from torchvision import transforms
from utils import generate_crop
import copy


class MoeDataset(Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform
        self.flip = transforms.Compose([transforms.RandomHorizontalFlip()])

    def __len__(self):
        return len(self.img_list)

    def rotate(self, img):
        times = random.randint(0, 3)
        img = np.rot90(img, k=times)
        return img, times

    def norm(self, img, percentage_low, percentage_high):
        img = (img - np.percentile(img, percentage_low)) / (
                np.percentile(img, percentage_high) - np.percentile(img, percentage_low) + 1e-7)
        img[img > 1] = 1
        img[img < 0] = 0
        return img

    def __getitem__(self, item):
        img = self.img_list[item]
        if isinstance(img, str):
            if img.endswith('.tif'):
                img = Image.fromarray(imread(img).astype(np.float))

            elif img.endswith('.png'):
                img = Image.open(img)
            else:
                print(img)
                raise Exception('file format not supported')
        else:
            img = Image.fromarray(img)
        width, height = img.size
        if width <= 256:
            img = img.resize((224, 224))
            img = self.flip(img)
        else:
            img = self.transform(img)
            img = np.array(img)
        img, target = self.rotate(img)
        img = self.norm(img, 2, 99)
        img = np.expand_dims(img, axis=0)
        # print(img.shape)
        return torch.tensor(img.copy()), torch.tensor(target)


class MoeAUDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list
        # self.transform = transform
        # self.flip = transforms.Compose([transforms.RandomHorizontalFlip()])

    def __len__(self):
        return len(self.img_list)


    def norm(self, img, percentage_low, percentage_high):
        img = (img - np.percentile(img, percentage_low)) / (
                np.percentile(img, percentage_high) - np.percentile(img, percentage_low) + 1e-7)
        img[img > 1] = 1
        img[img < 0] = 0
        return img

    def __getitem__(self, item):
        img = self.img_list[item]
        if isinstance(img, str):
            if img.endswith('.tif'):
                img = Image.fromarray(imread(img).astype(np.float))

            elif img.endswith('.png'):
                img = Image.open(img)
            else:
                print(img)
                raise Exception('file format not supported')
        else:
            img = Image.fromarray(img)
        width, height = img.size
        img = img.resize((224, 224))
        # if width <= 256:
        #     img = img.resize((224, 224))
        #     img = self.flip(img)
        # else:
        #     img = self.transform(img)
        #     img = np.array(img)
        # img, target = self.rotate(img)
        img = self.norm(img, 2, 99)
        img = np.expand_dims(img, axis=0)
        target = copy.deepcopy(img)
        # print(img.shape)
        return torch.tensor(img.copy()), torch.tensor(target)


def build_moe_dataset(args, data_root):
    print(data_root)
    img_list = []
    for file in os.listdir(os.path.join(data_root, 'training_input')):
        img_list.append(os.path.join(data_root, 'training_input', file))
    for file in os.listdir(os.path.join(data_root, 'validate_input')):
        img_list.append(os.path.join(data_root, 'validate_input', file))
    # determine the scale the randomReiszedCrop
    img_list = generate_crop(img_list)
    transform_list = [transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                      transforms.RandomHorizontalFlip()]  # todo how to deal with vairous size(some are big and some are small)
    # transform_list = [transforms.Resize(224)]
    transform_list = transforms.Compose(transform_list)
    dataset = MoeDataset(img_list, transform_list)
    dataloader = DataLoader(dataset, batch_size=args.b, num_workers=args.workers, shuffle=True)
    return dataloader


def moe_au_dataset(args, data_root):
    print(data_root)
    img_list = []
    for file in os.listdir(os.path.join(data_root, 'training_input')):
        img_list.append(os.path.join(data_root, 'training_input', file))
    for file in os.listdir(os.path.join(data_root, 'validate_input')):
        img_list.append(os.path.join(data_root, 'validate_input', file))
    # determine the scale the randomReiszedCrop
    img_list = generate_crop(img_list, patch_size=256, stride=128)
    dataset = MoeAUDataset(img_list)
    dataloader = DataLoader(dataset, batch_size=args.b, num_workers=args.workers, shuffle=True)
    return dataloader


class SelectDataset(Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def rotate(self, img, times):
        img = np.rot90(img, k=times)
        return img, times

    def norm(self, img, percentage_low, percentage_high):
        img = (img - np.percentile(img, percentage_low)) / (
                np.percentile(img, percentage_high) - np.percentile(img, percentage_low) + 1e-7)
        img[img > 1] = 1
        img[img < 0] = 0
        return img

    def __getitem__(self, item):
        img = self.img_list[item]
        if isinstance(img, str):
            if img.endswith('.tif'):
                img = Image.fromarray(imread(img).astype(np.float))

            elif img.endswith('.png'):
                img = Image.open(img)
            else:
                print(img)
                raise Exception('file format not supported')
        else:
            img = Image.fromarray(img)
        width, height = img.size

        img = self.transform(img)
        img = np.array(img)
        img_out = []
        target_out = []
        for i in range(4):
            img_, target = self.rotate(img, i)
            img_ = self.norm(img_, 2, 99)
            img_ut.append(img_)
            target_out.append(target)
        img_out = np.stack(img_out, axis=0)
        target_out = np.array(target_out)
        # img = np.expand_dims(img, axis=0)
        # print(img.shape)
        return torch.tensor(img_out.copy()), torch.tensor(target_out)
