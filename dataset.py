from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from utils import *
from tifffile import imread
from PIL import Image
import numpy as np
import torch.distributed as dist
import cv2
import random
import pickle
import torch.nn.functional as F
import copy


def my_collate_fn(batch):
    scale = [item[2] for item in batch]
    scale = torch.stack(scale, dim=0)
    # print(scale)
    # index1 = (scale == 1)
    # index2 = (scale == 2)
    # index3 = (scale == 3)
    # index4 = (scale == 4)
    gt_scale1 = []
    gt_scale2 = []
    gt_scale3 = []
    gt_scale4 = []
    img_scale1 = []
    img_scale2 = []
    img_scale3 = []
    img_scale4 = []
    for i in range(len(batch)):
        if scale[i] == 1:
            # print('scale1', batch[i][1].shape, batch[i][0].shape)
            gt_scale1.append(batch[i][1])
            img_scale1.append(batch[i][0])
        elif scale[i] == 2:
            # print('scale2', batch[i][1].shape, batch[i][0].shape)
            gt_scale2.append(batch[i][1])
            img_scale2.append(batch[i][0])
        elif scale[i] == 3:
            # print('scale3', batch[i][1].shape, batch[i][0].shape)
            gt_scale3.append(batch[i][1])
            img_scale3.append(batch[i][0])
        else:
            # print('scale4', batch[i][1].shape, batch[i][0].shape)
            gt_scale4.append(batch[i][1])
            img_scale4.append(batch[i][0])

    if len(gt_scale1) != 0:
        gt_scale1 = torch.stack(gt_scale1, dim=0)
        img_scale1 = torch.stack(img_scale1, dim=0)
    else:
        gt_scale1 = None
        img_scale1 = None
    if len(gt_scale2) != 0:
        gt_scale2 = torch.stack(gt_scale2, dim=0)
        img_scale2 = torch.stack(img_scale2, dim=0)
    else:
        gt_scale2 = None
        img_scale2 = None
    if len(gt_scale3) != 0:
        gt_scale3 = torch.stack(gt_scale3, dim=0)
        img_scale3 = torch.stack(img_scale3, dim=0)
    else:
        gt_scale3 = None
        img_scale3 = None
    if len(gt_scale4) != 0:
        gt_scale4 = torch.stack(gt_scale4, dim=0)
        img_scale4 = torch.stack(img_scale4, dim=0)
    else:
        gt_scale4 = None
        img_scale4 = None
    return [img_scale1, img_scale2, img_scale3, img_scale4], [gt_scale1, gt_scale2, gt_scale3, gt_scale4], scale


class DESUDataset(Dataset):
    def __init__(self, img_list, gt_list, train=False):
        self.train = train
        self.img_list = img_list
        self.gt_list = gt_list
        # self.resize_transforms = [transforms.Compose([transforms.Resize(224, 224)])]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img = self.img_list[item]
        gt = self.gt_list[item]
        if img.endswith('.tif'):
            img = imread(img).astype(np.float)
            gt = imread(gt).astype(np.float)
        elif img.endswith('.png'):
            img = np.array(Image.open(img))
            gt = np.array(Image.open(gt))
        else:
            raise Exception('file format not supported')
        if len(img.shape) == 3:
            img = img[0]
            gt = gt[0]
        scale = gt.shape[0] // img.shape[0]
        img, gt = self.aug(img, gt, scale)
        return torch.from_numpy(img).unsqueeze_(dim=0), torch.from_numpy(gt).unsqueeze_(dim=0), torch.tensor(scale)

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

    def aug(self, img, gt, scale):
        if img.shape[0] <= 128:
            # print(img.shape, gt.shape, scale)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (128 * scale, 128 * scale), interpolation=cv2.INTER_LINEAR)
            # print('after reshape;', img.shape, gt.shape, scale)
        else:
            img, gt = self.randomCrop(img, gt, size=128, scale=scale)
        if self.train:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            rot90 = random.random() < 0.5
            if hflip:
                img = cv2.flip(img, 1)
                gt = cv2.flip(gt, 1)
            if vflip:
                img = cv2.flip(img, 0)
                gt = cv2.flip(gt, 0)
            if rot90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
        img = self.norm(img, percentage_low=2, percentage_high=99)
        gt = self.norm(gt, percentage_low=2, percentage_high=99)
        return img, gt


def supervised_dataloader(args):

    train_img_list, train_gt_list = get_train_list(args.data,args.finetune)
    val_img_list, val_gt_list = get_valid_list(args.data, args.finetune)
    train_img_list = train_img_list + val_img_list
    train_gt_list = train_gt_list + val_gt_list
    # append the train dataset
    # train_img_list = train_img_list + train_img_list
    # train_gt_list = train_gt_list + train_gt_list
    train_dataset = DESUDataset(train_img_list, train_gt_list, train=True)
    val_dataset = DESUDataset(val_img_list, val_gt_list, train=False)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=dist.get_rank()
    # )
    # val_sampler = torch.utils.data.distributed.DistributedSampler(
    #     val_dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=dist.get_rank(),
    # )
    train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_worker, batch_size=args.b, pin_memory=True,
                              collate_fn=my_collate_fn, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_worker, batch_size=64, pin_memory=True,
                            collate_fn=my_collate_fn)

    return train_loader, val_loader


def supervised_selected_data_loader(args):
    with open(args.pickle_file, 'rb') as f:
        weight_dict = pickle.load(f)
    weight_dict_softed = copy.deepcopy(weight_dict)
    values = []
    for key, value in weight_dict.items():
        values.append(value)
    values = F.softmax(torch.tensor(values))
    for i, (key, value) in enumerate(weight_dict.items()):
        weight_dict_softed[key] = values[i]
    train_img_list, train_gt_list, train_weight_values = get_selected_train_list(args.data, args.finetune, weight_dict_softed)
    val_img_list, val_gt_list, val_weight_values = get_selected_train_list(args.data, args.finetune, weight_dict_softed)
    train_img_list = train_img_list + val_img_list
    train_weight_values = train_weight_values + val_weight_values
    train_gt_list = train_gt_list + val_gt_list

    idx = [i for i in range(len(train_img_list))]
    selected_idx = random.choices(idx, weights = train_weight_values, k=args.sample_num)
    selected_train_img = []
    selected_gt_img = []
    for i in selected_idx:
        selected_train_img.append(train_img_list[i])
        selected_gt_img.append(train_gt_list[i])
    print(f'total source data {len(train_img_list)} and we select {len(selected_train_img)}')
    train_dataset = DESUDataset(selected_train_img, selected_gt_img, train=True)
    val_dataset = DESUDataset(selected_train_img[int(len(selected_train_img) * 0.9):], selected_gt_img[int(len(selected_train_img) * 0.9):], train=False)
    train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_worker, batch_size=args.b, pin_memory=True,
                              collate_fn=my_collate_fn, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_worker, batch_size=64, pin_memory=True,
                            collate_fn=my_collate_fn)

    return train_loader, val_loader
