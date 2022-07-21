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
    train_img_list, train_gt_list = get_train_list(args.data, args.finetune)
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
        if args.au:
            values.append(1 - value)
        else:
            values.append(value)
    values = F.softmax(torch.tensor(values))
    for i, (key, value) in enumerate(weight_dict.items()):
        weight_dict_softed[key] = values[i]
    train_img_list, train_gt_list, train_weight_values, train_scale_list = get_selected_train_list(args.data,
                                                                                                   args.finetune,
                                                                                                   weight_dict_softed)
    val_img_list, val_gt_list, val_weight_values, val_scale_list = get_selected_train_list(args.data, args.finetune,
                                                                                           weight_dict_softed)
    train_img_list = train_img_list + val_img_list
    train_weight_values = train_weight_values + val_weight_values
    train_scale_list = train_scale_list + val_scale_list
    train_gt_list = train_gt_list + val_gt_list
    if args.uniform:
        print('Performing Uniform sampling on dataset')
        train_weight_values = [1 / len(train_img_list) for _ in range(len(train_img_list))]
    idx = [i for i in range(len(train_img_list))]
    selected_idx = random.choices(idx, weights=train_weight_values, k=args.sample_num)
    selected_train_img = []
    selected_gt_img = []
    selected_scale = []
    for i in selected_idx:
        selected_train_img.append(train_img_list[i])
        selected_gt_img.append(train_gt_list[i])
        selected_scale.append(train_scale_list[i])
    print(f'total source data {len(train_img_list)} and we select {len(selected_train_img)}')
    sorted_img_list = [[] for _ in range(4)]
    sorted_gt_list = [[] for _ in range(4)]
    for idx in range(len(selected_train_img)):
        sorted_img_list[selected_scale[idx] - 1].append(selected_train_img[idx])
        sorted_gt_list[selected_scale[idx] - 1].append(selected_gt_img[idx])
    train_loaders = []
    for i in range(4):
        print(f'scale {i + 1} image length{len(sorted_img_list[i])}')
        train_dataset = DESUDataset(sorted_img_list[i], sorted_gt_list[i], train=True)
        batch_size = [args.b, args.b // 2, args.b // 3, args.b // 4]
        train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_worker, batch_size=int(batch_size[i]),
                                  pin_memory=True,
                                  shuffle=True)
        train_loaders.append(train_loader)

    val_dataset = DESUDataset(selected_train_img[int(len(selected_train_img) * 0.9):],
                              selected_gt_img[int(len(selected_train_img) * 0.9):], train=False)
    val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_worker, batch_size=64, pin_memory=True,
                            collate_fn=my_collate_fn)
    return train_loaders, val_loader


def softmax_dict(weight_dict, args):
    weight_dict_softed = copy.deepcopy(weight_dict)
    values = []
    for key, value in weight_dict.items():
        if args.au:
            values.append(1 - value)
        else:
            values.append(value)
    values = F.softmax(torch.tensor(values))
    for i, (key, value) in enumerate(weight_dict.items()):
        weight_dict_softed[key] = values[i]
    return weight_dict_softed


def select_data(train_img_list, train_weight_values, train_scale_list, train_gt_list, sample_num):
    idx = [i for i in range(len(train_img_list))]
    selected_idx = random.choices(idx, weights=train_weight_values, k=sample_num)
    selected_train_img = []
    selected_gt_img = []
    selected_scale = []
    for i in selected_idx:
        selected_train_img.append(train_img_list[i])
        selected_gt_img.append(train_gt_list[i])
        selected_scale.append(train_scale_list[i])
    print(f'total source data {len(train_img_list)} and we select {len(selected_train_img)}')
    sorted_img_list = [[] for _ in range(4)]
    sorted_gt_list = [[] for _ in range(4)]
    for idx in range(len(selected_train_img)):
        sorted_img_list[selected_scale[idx] - 1].append(selected_train_img[idx])
        sorted_gt_list[selected_scale[idx] - 1].append(selected_gt_img[idx])
    return selected_train_img, selected_gt_img, sorted_img_list, sorted_gt_list


def supervised_selected_rotNet_data_loader(args):
    with open(args.pickle_file, 'rb') as f:
        weight_dict = pickle.load(f)
    with open(args.pickle_file_mid, 'rb') as f:
        weight_dict_mid = pickle.load(f)
    weight_dict_softed = softmax_dict(weight_dict, args)
    weight_dict_mid_softed = softmax_dict(weight_dict_mid, args)
    train_img_list, train_gt_list, train_weight_values, train_scale_list = get_selected_rotNet_train_list(args.data,
                                                                                                          args.finetune,
                                                                                                          weight_dict_softed)
    val_img_list, val_gt_list, val_weight_values, val_scale_list = get_selected_rotNet_train_list(args.data,
                                                                                                  args.finetune,
                                                                                                  weight_dict_softed,
                                                                                                  train=False)
    train_img_list_mid, train_gt_list_mid, train_weight_values_mid, train_scale_list_mid = get_selected_rotNet_train_list(
        args.data,
        args.finetune,
        weight_dict_mid_softed, mid=True)
    val_img_list_mid, val_gt_list_mid, val_weight_values_mid, val_scale_list_mid = get_selected_rotNet_train_list(
        args.data,
        args.finetune,
        weight_dict_mid_softed,
        train=False, mid=True)
    train_img_list = train_img_list + val_img_list
    train_weight_values = train_weight_values + val_weight_values
    train_scale_list = train_scale_list + val_scale_list
    train_gt_list = train_gt_list + val_gt_list
    print('train img list:', len(train_img_list), sum(train_weight_values))
    selected_train_img, selected_gt_img, sorted_img_list, sorted_gt_list = select_data(train_img_list,
                                                                                       train_weight_values,
                                                                                       train_scale_list, train_gt_list,
                                                                                       sample_num=14000)

    train_img_list = train_img_list_mid + val_img_list_mid
    train_weight_values = train_weight_values_mid + val_weight_values_mid
    train_scale_list = train_scale_list_mid + val_scale_list_mid
    train_gt_list = train_gt_list_mid + val_gt_list_mid

    selected_train_img_mid, selected_gt_img_mid, sorted_img_list_mid, sorted_gt_list_mid = select_data(train_img_list,
                                                                                                       train_weight_values,
                                                                                                       train_scale_list,
                                                                                                       train_gt_list,
                                                                                                       sample_num=16000)
    train_loaders = []
    selected_train_img = selected_train_img + selected_train_img_mid
    selected_gt_img = selected_gt_img + selected_gt_img_mid
    for i in range(4):
        print(f'scale {i + 1} image length{len(sorted_img_list[i])}')
        train_dataset = DESUDataset(sorted_img_list[i] + sorted_img_list_mid[i],
                                    sorted_gt_list[i] + sorted_gt_list_mid[i], train=True)
        batch_size = [args.b, args.b // 2, args.b // 3, args.b // 4]
        train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_worker, batch_size=int(batch_size[i]),
                                  pin_memory=True,
                                  shuffle=True)
        train_loaders.append(train_loader)

    val_dataset = DESUDataset(selected_train_img[int(len(selected_train_img) * 0.9):],
                              selected_gt_img[int(len(selected_train_img) * 0.9):], train=False)
    val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_worker, batch_size=64, pin_memory=True,
                            collate_fn=my_collate_fn)
    return train_loaders, val_loader


def supervised_selected_mmd_data_loader(args):
    with open(args.pickle_file, 'rb') as f:
        weight_dict = pickle.load(f)
    finetune_key = args.finetune
    mmd_values = dict()
    for tissue_key in weight_dict.keys():
        print(tissue_key)

        if finetune_key in tissue_key:
            # print(tissue_key)
            # other_data = tissue_key.split('to')
            # print(other_data)
            # other_data = other_data[0] if other_data[1] == finetune_key else other_data[1]
            mmd_values[tissue_key] = weight_dict[tissue_key]
    print(len(mmd_values.values()))
    mmd_values_sorted = sorted(mmd_values.values())
    print(mmd_values_sorted)
    threshhold = mmd_values_sorted[int(len(mmd_values.values()) * 0.5)]
    print(threshhold, mmd_values_sorted[0], mmd_values_sorted[-1])
    img_list, gt_list, scale_list = [], [], []
    for sub_dir in os.listdir(args.data):
        for tissue in os.listdir(os.path.join(args.data, sub_dir)):
            key = os.path.join(sub_dir, tissue)
            if key == args.finetune:
                continue
            try:
                mmd_vlaue = mmd_values[args.finetune + 'to' + key]
            except:
                mmd_vlaue = mmd_values[key + 'to' + args.finetune]
            if mmd_vlaue > threshhold:
                continue
            for prefix in ['training', 'validate']:
                tissue_path = os.path.join(args.data, sub_dir, tissue)
                if len(os.listdir(os.path.join(tissue_path, prefix + '_input'))) == 0:
                    continue
                # print(os.path.join(tissue_path, prefix+'_input'))
                sample_img = os.path.join(tissue_path, prefix + '_input',
                                          os.listdir(os.path.join(tissue_path, prefix + '_input'))[0])
                sample_gt = os.path.join(tissue_path, prefix + '_gt', os.path.basename(sample_img))
                if sample_img.endswith('.tif'):
                    sample_img = imread(sample_img).astype(np.float)
                    sample_gt = imread(sample_gt).astype(np.float)

                elif sample_img.endswith('.png'):
                    sample_img = np.array(Image.open(sample_img))
                    sample_gt = np.array(Image.open(sample_gt))
                (h, w) = np.shape(sample_img)
                (hr, wr) = np.shape(sample_gt)
                scale = hr // h
                for file in os.listdir(os.path.join(tissue_path, prefix + '_input')):
                    img_list.append(os.path.join(tissue_path, prefix + '_input', file))
                    scale_list.append(scale)
                    assert os.path.exists(os.path.join(tissue_path, prefix + '_gt', file)), print(
                        os.path.join(tissue_path, prefix + '_gt', file))
                    gt_list.append(os.path.join(tissue_path, prefix + '_gt', file))
    sorted_img_list = [[] for _ in range(4)]
    sorted_gt_list = [[] for _ in range(4)]
    for idx in range(int(len(img_list) * 0.9)):
        sorted_img_list[scale_list[idx] - 1].append(img_list[idx])
        sorted_gt_list[scale_list[idx] - 1].append(gt_list[idx])
    train_loaders = []
    for i in range(4):
        print(f'scale {i + 1} image length{len(sorted_img_list[i])}')
        if len(sorted_img_list[i]) == 0:
            train_loaders.append(None)
            continue
        train_dataset = DESUDataset(sorted_img_list[i], sorted_gt_list[i], train=True)
        batch_size = [args.b, args.b // 2, args.b // 3, args.b // 4]
        train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_worker, batch_size=int(batch_size[i]),
                                  pin_memory=False,
                                  shuffle=True)
        train_loaders.append(train_loader)

    val_dataset = DESUDataset(img_list[int(len(img_list) * 0.9):],
                              gt_list[int(len(gt_list) * 0.9):], train=False)
    val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_worker, batch_size=64, pin_memory=False,
                            collate_fn=my_collate_fn)
    return train_loaders, val_loader


def get_similar_list(root_dir, finetune_data, similar, tissue='actin'):
    img_list = []
    gt_list = []
    if tissue == 'actin':
        similar_subdir = ['1_processed', '5_processed', '7', '10_processed', '12_processed', '']
        similar_tissue = ['WideField_BPAE_G', 'F-actin ', 'F-actin-nonlinear', 'mitochondria', 'actin', 'actin-20x-noise1', 'actin-60x-noise1', 'actin-60x-noise2', 'actin-confocal', 'Confocal_BPAE_G', 'TwoPhoton_BPAE_G']
    else:
        print('select mito data')
        similar_subdir = [ '5_processed', '2_processed', '12_processed', '10_processed']
        similar_tissue = ['mitochondria', 'mitotracker','neuronalMito'
                          'mito-confocal', 'mito-60x-noise2', 'mito-60x-noise1', 'mito-20x-noise1', 'Confocal_BPAE_R', 'TwoPhoton_BPAE_R']
    for subdir in os.listdir(root_dir):

        subdir_path = os.path.join(root_dir, subdir)
        for tissue in os.listdir(subdir_path):
            if subdir == finetune_data.split('/')[0] and tissue == finetune_data.split('/')[1]:
                print('skiping')
                continue
            tissue_path = os.path.join(subdir_path, tissue)
            if similar == False:
                if subdir in similar_subdir and tissue in similar_tissue:
                    continue
                for file in os.listdir(os.path.join(tissue_path, 'training_input')):
                    img_list.append(os.path.join(tissue_path, 'training_input', file))
                    assert os.path.exists(os.path.join(tissue_path, 'training_gt', file)), print(
                        os.path.join(tissue_path, 'training_gt', file))
                    gt_list.append(os.path.join(tissue_path, 'training_gt', file))
                for file in os.listdir(os.path.join(tissue_path, 'validate_input')):
                    img_list.append(os.path.join(tissue_path, 'validate_input', file))
                    assert os.path.exists(os.path.join(tissue_path, 'validate_gt', file)), print(
                        os.path.join(tissue_path, 'validate_gt', file))
                    gt_list.append(os.path.join(tissue_path, 'validate_gt', file))
            else:
                if subdir in similar_subdir and tissue in similar_tissue:
                    for file in os.listdir(os.path.join(tissue_path, 'training_input')):
                        img_list.append(os.path.join(tissue_path, 'training_input', file))
                        assert os.path.exists(os.path.join(tissue_path, 'training_gt', file)), print(
                            os.path.join(tissue_path, 'training_gt', file))
                        gt_list.append(os.path.join(tissue_path, 'training_gt', file))
                    for file in os.listdir(os.path.join(tissue_path, 'validate_input')):
                        img_list.append(os.path.join(tissue_path, 'validate_input', file))

                        assert os.path.exists(os.path.join(tissue_path, 'validate_gt', file)), print(
                            os.path.join(tissue_path, 'validate_gt', file))
                        gt_list.append(os.path.join(tissue_path, 'validate_gt', file))

    return img_list, gt_list


def supervised_select_similar_data_loader(args):
    train_img_list, train_gt_list = get_similar_list(args.data, args.finetune, similar=False, tissue=args.simtissue)
    train_weight_values = [1 / len(train_img_list) for _ in range(len(train_img_list))]
    idx = [i for i in range(len(train_img_list))]
    if args.zero:
        selected_idx = random.choices(idx, weights=train_weight_values, k=int(len(train_img_list) * 0.0))
    else:
        selected_idx = random.choices(idx, weights=train_weight_values, k=int(len(train_img_list) * 0.3))
    selected_train_img = []
    selected_gt_img = []
    for i in selected_idx:
        selected_train_img.append(train_img_list[i])
        selected_gt_img.append(train_gt_list[i])
    similar_imgs, similar_gts = get_similar_list(args.data, args.finetune, similar=True)
    print(
        f'total source data {len(train_img_list)} and we select {len(selected_train_img)} and similar imgs have {len(similar_imgs)}')
    selected_train_img.extend(similar_imgs)
    selected_gt_img.extend(similar_gts)
    index = [i for i in range(len(selected_train_img))]
    from random import shuffle
    shuffle(index)
    train_imgs = []
    train_gts = []
    val_imgs = []
    val_gts = []

    for i in range(len(index)):
        if i < len(index) * 0.9:
            train_imgs.append(selected_train_img[index[i]])
            train_gts.append(selected_gt_img[index[i]])
        else:
            val_imgs.append(selected_train_img[index[i]])
            val_gts.append(selected_gt_img[index[i]])

    train_dataset = DESUDataset(train_imgs, train_gts, train=True)
    val_dataset = DESUDataset(val_imgs, val_gts, train=False)
    train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_worker, batch_size=args.b, pin_memory=True,
                              collate_fn=my_collate_fn, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_worker, batch_size=64, pin_memory=True,
                            collate_fn=my_collate_fn)

    return train_loader, val_loader

def supervised_select_dissimilar_data_loader(args):
    train_img_list, train_gt_list = get_similar_list(args.data, args.finetune, similar=False, tissue=args.simtissue)
    train_weight_values = [1 / len(train_img_list) for _ in range(len(train_img_list))]
    idx = [i for i in range(len(train_img_list))]

    similar_imgs, similar_gts = get_similar_list(args.data, args.finetune, similar=True)
    if args.zero:
        selected_idx = random.choices(idx, weights=train_weight_values,
                                      k=int(len(train_img_list) * 0.0) + len(similar_imgs))
    else:
        selected_idx = random.choices(idx, weights=train_weight_values, k=int(len(train_img_list) * 0.3) + len(similar_imgs))
    selected_train_img = []
    selected_gt_img = []
    for i in selected_idx:
        selected_train_img.append(train_img_list[i])
        selected_gt_img.append(train_gt_list[i])
    # selected_train_img.extend(similar_imgs)
    # selected_gt_img.extend(similar_gts)
    print(
        f'total source data {len(train_img_list)} and we select {len(selected_train_img)} and similar imgs have 0')
    index = [i for i in range(len(selected_train_img))]
    from random import shuffle
    shuffle(index)
    train_imgs = []
    train_gts = []
    val_imgs = []
    val_gts = []

    for i in range(len(index)):
        if i < len(index) * 0.9:
            train_imgs.append(selected_train_img[index[i]])
            train_gts.append(selected_gt_img[index[i]])
        else:
            val_imgs.append(selected_train_img[index[i]])
            val_gts.append(selected_gt_img[index[i]])

    train_dataset = DESUDataset(train_imgs, train_gts, train=True)
    val_dataset = DESUDataset(val_imgs, val_gts, train=False)
    train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_worker, batch_size=args.b, pin_memory=True,
                              collate_fn=my_collate_fn, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_worker, batch_size=64, pin_memory=True,
                            collate_fn=my_collate_fn)

    return train_loader, val_loader