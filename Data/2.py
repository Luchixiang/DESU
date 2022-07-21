# import tifffile
import imageio
import numpy as np
import os
from skimage import io
import shutil
import tifffile as tiff

# path = '/Users/luchixiang/Downloads/disk/SR_Data/2/mitotracker_PSSR-SF/lr/train/mitotracker/mitotracker_PSSR-SF_train_lr_0002.tif'
# # path_hr = '/Users/luchixiang/Downloads/disk/SR_Data/2/mitotracker_PSSR-SF/hr/train/mitotracker/mitotracker_PSSR-SF_train_hr_0002.tif'
# # path_3 = '/Users/luchixiang/Downloads/disk/SR_Data/2/neuronalMito_PSSR-MF/lr/train/neurons/neuronalMito_PSSR-MF_train_lr_0001.npy'
# # path_4 = '/Users/luchixiang/Downloads/disk/SR_Data/2/neuronalMito_PSSR-MF/hr/train/neurons/neuronalMito_PSSR-MF_train_hr_0001.tif'
# # a = imageio.imread(path_4)
# # a = np.array(a)
# # c = np.load(path_3)
# # print(c.shape)
# # print(a.shape)
base_path = '/mnt/sdb/cxlu/SR_Data/2/training/trainsets/neuronalMito_PSSR-MF/lr/train/neurons'
dist_base_path = '/mnt/sdb/cxlu/SR_Data_processed/2_processed/neuronalMito'
# training:
os.mkdir(dist_base_path)
os.mkdir(os.path.join(dist_base_path, 'training_input'))
os.mkdir(os.path.join(dist_base_path, 'training_gt'))
os.mkdir(os.path.join(dist_base_path, 'validate_input'))
os.mkdir(os.path.join(dist_base_path, 'validate_gt'))
for file in os.listdir(base_path):
    npy_file = os.path.join(base_path, file)
    image = np.load(npy_file)[2]
    dist_file = os.path.join(dist_base_path, 'training_input', file.replace('.npy', '.tif'))
    gt_file = os.path.join(base_path.replace('lr', 'hr'), file.replace('lr', 'hr').replace('.npy', '.tif'))
    tiff.imsave(dist_file, image)
    shutil.copy(gt_file, os.path.join(dist_base_path, 'training_gt', file.replace('.npy', '.tif')))
    print(dist_file, os.path.join(dist_base_path, 'training_gt', file.replace('.npy', '.tif')))
base_valid_path = '/mnt/sdb/cxlu/SR_Data/2/training/trainsets/neuronalMito_PSSR-MF/lr/valid/neurons'
for file in os.listdir(base_valid_path):
    npy_file = os.path.join(base_valid_path, file)
    image = np.load(npy_file)[2]
    dist_file = os.path.join(dist_base_path,'validate_input', file.replace('.npy', '.tif'))
    tiff.imsave(dist_file, image)
    gt_file = os.path.join(base_valid_path.replace('lr', 'hr'), file.replace('lr', 'hr').replace('.npy', '.tif'))
    shutil.copy(gt_file, os.path.join(dist_base_path, 'validate_gt', file.replace('.npy', '.tif')))
    print(dist_file)

base_path = '/mnt/sdb/cxlu/SR_Data/2/training/trainsets/mitotracker_PSSR-SF/lr/train/mitotracker'
base_valid_path = '/mnt/sdb/cxlu/SR_Data/2/training/trainsets/mitotracker_PSSR-SF/lr/valid/mitotracker'
dist_base_path = '/mnt/sdb/cxlu/SR_Data_processed/2_processed/mitotracker'
os.mkdir(dist_base_path)
os.mkdir(os.path.join(dist_base_path, 'training_input'))
os.mkdir(os.path.join(dist_base_path, 'training_gt'))
os.mkdir(os.path.join(dist_base_path, 'validate_input'))
os.mkdir(os.path.join(dist_base_path, 'validate_gt'))
for file in os.listdir(base_path):
    gt_file = os.path.join(base_path.replace('lr', 'hr'), file.replace('lr', 'hr'))
    shutil.copy(os.path.join(base_path, file), os.path.join(dist_base_path, 'training_input', file))
    shutil.copy(gt_file, os.path.join(dist_base_path, 'training_gt', file))
for file in os.listdir(base_valid_path):
    gt_file = os.path.join(base_valid_path.replace('lr', 'hr'), file.replace('lr', 'hr'))
    shutil.copy(os.path.join(base_valid_path, file), os.path.join(dist_base_path, 'validate_input', file))
    shutil.copy(gt_file, os.path.join(dist_base_path, 'validate_gt', file))

