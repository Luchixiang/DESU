import os
import random
from tifffile import imread
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
from numpy.linalg import norm
import pickle
from sklearn.metrics.pairwise import cosine_similarity


def generate_crop(img_list, patch_size=256, stride=256):
    # width, height = img.size
    sample_img = img_list[0]
    # print(sample_img)
    if sample_img.endswith('.tif'):
        img = imread(sample_img)
        width, height = img.shape
    elif sample_img.endswith('.png'):
        img = Image.open(sample_img)
        width, height = img.size
    # print(f'sample image size{width}, {height}')
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
            left = (w - patch_size) / 2
            top = (h - patch_size) / 2
            right = (w + patch_size) / 2
            bottom = (h + patch_size) / 2
            img = Image.fromarray(img)
            img = img.crop((left, top, right, bottom))
            cropped_list.append(np.array(img))
        return cropped_list
    else:
        np_list = []
        for img in img_list:
            if img.endswith('.tif'):
                img = imread(img).astype(np.float)
            elif img.endswith('.png'):
                img = np.array(Image.open(img))
                (h, w) = np.shape(img)
            img = Image.fromarray(img)
            img = img.resize((224, 224))
            np_list.append(np.array(img))
        return np_list


def norm(img):
    img_min = np.percentile(img, 2)
    img_max = np.percentile(img, 98)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    img[img < 0] = 0
    img[img > 1] = 1
    return img
    # img = self.norm(img, percentage_low=2, percentage_high=99)


def cal_sim_between_two_dataset(source_img_path, target_img_path):
    source_example_imgs = dict()
    for tissue in os.listdir(source_img_path):
        source_example_imgs[tissue] = []
        tissue_path = os.path.join(source_img_path, tissue)
        for img in os.listdir(os.path.join(tissue_path)):
            if 'tif' not in img:
                continue
            index = img.split('_')[-1]
            if 'num' in img and int(index[3:-4]) < 10:
                source_example_imgs[tissue].append(os.path.join(tissue_path, img))

    target_imgs = []
    tissue_path = target_img_path
    for img in os.listdir(os.path.join(tissue_path, 'training_input')):
        target_imgs.append(os.path.join(tissue_path, 'training_input', img))

    selected_target_imgs = random.sample(target_imgs, k=10)
    selected_target_imgs = generate_crop(selected_target_imgs)
    print(len(selected_target_imgs), len(source_example_imgs.keys()))
    sims = []
    sim_dict = dict()
    for tissue in sorted(source_example_imgs.keys()):
        tissue_sim = 0.0
        for target_img in selected_target_imgs:
            target_img = cv2.resize(target_img, (224, 224))
            target_img = norm(target_img)
            source_imgs = source_example_imgs[tissue]
            for img in source_imgs:
                sr_img = imread(img)
                sr_img = norm(sr_img)
                tissue_sim += structural_similarity(target_img, sr_img, data_range=1)
        tissue_sim /= (len(selected_target_imgs) * len(source_example_imgs[tissue]))

        sims.append(tissue_sim)
        sim_dict[tissue] = tissue_sim
    return sims, sim_dict


if __name__ == '__main__':
    source_root = '/mnt/sdb/cxlu/SR_Data_processed'
    middle_dataset_path = './DatasetCondensation/10_result'
    out_dict = dict()
    target_dataset_path = '/mnt/sdb/cxlu/SR_Data_processed/16_denoise/tissue'
    tar_mid_sims, tar_mid_simdict = cal_sim_between_two_dataset(source_img_path=middle_dataset_path,
                                                                target_img_path=target_dataset_path)
    print(f'tar_mid_sims{tar_mid_sims}')
    for subdir in os.listdir(source_root):
        if '10_' in subdir:  # middle
            continue
        if '16_' in subdir:  # test dataset
            continue
        for tissue in os.listdir(os.path.join(source_root, subdir)):
            tissue_path = os.path.join(source_root, subdir, tissue)
            sims, _ = cal_sim_between_two_dataset(source_img_path=middle_dataset_path, target_img_path=tissue_path)
            print(sims)
            consine_sim = cosine_similarity([sims], [tar_mid_sims])[0][0]
            print(consine_sim)
            out_dict[tissue_path] = consine_sim
    print(out_dict.keys())
    with open('select_rotNet_16.pkl', 'wb') as f:
        pickle.dump(out_dict, f)
    with open('select_rotNet_16_mid.pkl', 'wb') as f:
        pickle.dump(tar_mid_simdict, f)
