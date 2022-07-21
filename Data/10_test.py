import os
import shutil
from PIL import Image
import numpy as np
path = '/mnt/sdb/cxlu/SR_Data/10/fmdd'
dist_path = '/mnt/sdb/cxlu/SR_Data/10_processed'
all_noise_levels = [1, 2, 4, 8, 16]
all_types = ['test_mix']
fovs = list(range(1, 20+1))
type_path = os.path.join(path, all_types[0])
clean_path = os.path.join(path, all_types[0], 'gt')
for noise_level in all_noise_levels:
    if noise_level == 1:
        noise_dir = os.path.join(type_path, 'raw')
    else:
        noise_dir = os.path.join(type_path, f'avg{noise_level}')

    for file in os.listdir(noise_dir):
        clean_file = os.path.join(clean_path, file)
        if '.png' in file:
            fov = file.split('_')[-1][:-4]
            type = file.split('_')[:-1]
            type = '_'.join(type)
            a = Image.open(clean_file)
            b = Image.open(os.path.join(noise_dir, file))
            a = np.array(a)
            b = np.array(b)
            print(a.shape, b.shape, a.max())
            shutil.copy(os.path.join(noise_dir, file), os.path.join(dist_path, type, 'validate_gt', f'noise{noise_level}_fov{fov}_{file}'))
            shutil.copy(os.path.join(noise_dir, file), os.path.join(dist_path, type, 'validate_input', f'noise{noise_level}_fov{fov}_{file}'))
    #         print(os.path.join(noise_dir, file), os.path.join(dist_path, type, 'validate_gt', f'noise{noise_level}_fov{fov}_{file}'))
    # # for i_fov in fovs:
    #     noisy_fov_dir = os.path.join(noise_dir, f'{i_fov}')
    #     clean_file = os.path.join(gt_dir, f'{i_fov}', 'avg50.png')
    #     for file in os.listdir(noisy_fov_dir):
    #         if '.png' in file:
    #             shutil.copy(os.path.join(noisy_fov_dir, file), os.path.join(dist_path, subdir, 'training_input', f'noise{noise_level}_fov{i_fov}_{file}'))
    #             shutil.copy(clean_file, os.path.join(dist_path, subdir, 'training_gt', f'fov{i_fov}_{file}'))
    #             print(os.path.join(noisy_fov_dir, file), os.path.join(dist_path, subdir, 'training_input', f'noise{noise_level}_fov{i_fov}_{file}'))
    #
