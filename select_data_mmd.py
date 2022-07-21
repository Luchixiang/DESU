import os
import random
from tifffile import imread
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
from numpy.linalg import norm
from utils import MMD_Max

import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn


def Normalize(x):
    norm_x = x.pow(2).sum(1, keepdim=True).pow(1. / 2.)
    x = x.div(norm_x)
    return x


def generate_crop(img_list, patch_size=512, stride=256):
    sample_img = img_list[0]
    if sample_img.endswith('.tif'):
        img = Image.fromarray(imread(sample_img).astype(np.float64))
    elif sample_img.endswith('.png'):
        img = Image.open(sample_img)
    width, height = img.size
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
        cropped_list = []
        for img in img_list:
            if img.endswith('.tif'):
                img = imread(img).astype(np.float64)

            elif img.endswith('.png'):
                img = np.array(Image.open(img))
            cropped_list.append(img)
        return cropped_list


class MMDDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = generate_crop(img_list, patch_size=512, stride=256)
        # self.mean = np.mean(img_list)
        # self.std = np.std(img_list)


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img = self.img_list[item]
        img = cv2.resize(img, (224, 224))
        # img = (img - self.mean)/self.std
        img = self.norm(img)
        return torch.from_numpy(img).unsqueeze_(dim=0).repeat((3, 1, 1)).float()

    def norm(self, img):
        img_min = np.percentile(img, 1)
        img_max = np.percentile(img, 99)
        img = (img - img_min) / (img_max - img_min + 1e-7)
        img[img < 0] = 0
        img[img > 1] = 1
        return img


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    source_root = '/mnt/sdb/cxlu/SR_Data_processed'
    if not os.path.exists('mmd_features_layer-4.pkl'):
        model = models.vgg16(pretrained=True).cuda()
        moodel = nn.Sequential(*list(model.children())[:-4])
        model.eval()
        tissue_features = dict()
        for subdir in os.listdir(source_root):
            for tissue in os.listdir(os.path.join(os.path.join(source_root, subdir))):
                tissue_path = os.path.join(source_root, subdir, tissue)
                key = os.path.join(subdir, tissue)
                img_list = []
                for file in os.listdir(os.path.join(tissue_path, 'training_input')):
                    img_list.append(os.path.join(tissue_path, 'training_input', file))
                tissue_dataset = MMDDataset(img_list)
                tissue_loader = DataLoader(tissue_dataset, batch_size=64, num_workers=4, shuffle=False)
                features = []
                if len(tissue_dataset) == 1:
                    print('img size 1:', subdir, tissue, len(tissue_dataset))
                for idx, (img) in enumerate(tissue_loader):
                    img = img.cuda()
                    # print(img.shape)
                    with torch.no_grad():
                        feature = model(img)
                        feature = Normalize(feature)
                        # print(feature.shape)
                        features.append(feature)
                        # features.append()
                tissue_features[key] = features
                with open('mmd_features.pkl', 'wb') as f:
                    pickle.dump(tissue_features, f)
    else:
        with open('mmd_features_layer-4', 'rb') as f:
            tissue_features = pickle.load(f)
    tissue_keys = sorted(tissue_features.keys())
    out_dict = dict()
    for i in range(len(tissue_keys)):
        for j in range(i + 1, len(tissue_keys)):
            feature_x = torch.cat(tissue_features[tissue_keys[i]], dim=0)
            feature_y = torch.cat(tissue_features[tissue_keys[j]], dim=0)
            print(feature_x.shape, feature_y.shape)
            # print('out', feature_x.shape, feature_y.shape)
            if feature_x.shape[0] < feature_y.shape[0]:
                feature_x, feature_y = feature_y, feature_x
            multiple = feature_x.shape[0] // feature_y.shape[0]
            mmd = 0.0
            for m in range(multiple):
                mmd += MMD_Max(feature_x[m * feature_y.shape[0]: (m + 1) * feature_y.shape[0]], feature_y, 'multiscale')
            mmd /= multiple
            key = tissue_keys[i] + 'to' + tissue_keys[j]
            out_dict[key] = mmd.cpu().data
    with open('./mmd_layer-4.pkl', 'wb') as f:
        pickle.dump(out_dict, f)
