import os
import shutil
from PIL import Image
import numpy as np
path = '/mnt/sdb/cxlu/SR_Data/10/fmdd'
dist_path = '/mnt/sdb/cxlu/SR_Data/10_processed'
all_noise_levels = [1, 2, 4, 8, 16]
all_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
             'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
             'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',
             'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
fovs = list(range(1, 20+1))
for subdir in all_types:
    os.mkdir(os.path.join(dist_path, subdir))
    os.mkdir(os.path.join(dist_path, subdir, 'training_input'))
    os.mkdir(os.path.join(dist_path, subdir, 'training_gt'))
    os.mkdir(os.path.join(dist_path, subdir, 'validate_input'))
    os.mkdir(os.path.join(dist_path, subdir, 'validate_gt'))
    type_path = os.path.join(path, subdir)
    gt_dir = os.path.join(type_path, 'gt')
    for noise_level in all_noise_levels:
        if noise_level == 1:
            noise_dir = os.path.join(type_path, 'raw')
        else:
            noise_dir = os.path.join(type_path, f'avg{noise_level}')
        for i_fov in fovs:
            noisy_fov_dir = os.path.join(noise_dir, f'{i_fov}')
            clean_file = os.path.join(gt_dir, f'{i_fov}', 'avg50.png')
            for file in os.listdir(noisy_fov_dir):
                if '.png' in file:

                    shutil.copy(os.path.join(noisy_fov_dir, file), os.path.join(dist_path, subdir, 'training_input', f'noise{noise_level}_fov{i_fov}_{file}'))
                    shutil.copy(clean_file, os.path.join(dist_path, subdir, 'training_gt', f'noise{noise_level}_fov{i_fov}_{file}'))
                    print(os.path.join(noisy_fov_dir, file), os.path.join(dist_path, subdir, 'training_input', f'noise{noise_level}_fov{i_fov}_{file}'))

