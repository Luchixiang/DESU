import os
source_root = '/mnt/sdb/cxlu/SR_Data_processed'
from random import shuffle
import numpy as np
from PIL import Image
from tifffile import imread
import shutil
for subdir in os.listdir(source_root):
    for tissue in os.listdir(os.path.join(os.path.join(source_root, subdir))):
        tissue_path = os.path.join(source_root, subdir, tissue)
        key = os.path.join(subdir, tissue)
        img_list = []
        for file in os.listdir(os.path.join(tissue_path, 'training_input')):
            img_list.append(os.path.join(tissue_path, 'training_input', file))
        shuffle(img_list)
        for i in range(5):

            sample = img_list[i]

            if sample.endswith('npy'):
                name = subdir + '_' + tissue + '_' + os.path.basename(sample)[:-4] + '.png'
                image = Image.fromarray(np.load(sample)).convert("L")
                image.save(os.path.join('./sample', name))
            elif sample.endswith('tif'):
                name = subdir + '_' + tissue + '_' + os.path.basename(sample)

                shutil.copy(sample, os.path.join('./sample', name))
                name = subdir + '_' + tissue + '_' + os.path.basename(sample)[:-4] + '.png'
                image = imread(sample)
                image = Image.fromarray(image).convert("L")
                image.save(os.path.join('./sample', name))
            else:
                name = subdir + '_' + tissue + '_' + os.path.basename(sample)[:-4] + '.png'
                shutil.copy(sample, os.path.join('./sample', name))
