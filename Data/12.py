import os
from tifffile import imread, imwrite
import numpy as np
path = '/mnt/sdb/cxlu/SR_Data/12'
dist_path = '/mnt/sdb/cxlu/SR_Data_processed/12_processed/'
types = ['actin-20x-noise1', 'actin-60x-noise1', 'actin-60x-noise2', 'actin-confocal', 'membrane', 'mito-20x-noise1', 'mito-60x-noise1',
        'mito-60x-noise2', 'mito-confocal', 'nucleus']

# for file_name in os.listdir(path):
#     file = os.path.join(path, file_name)
#     a = imread(file)
#     print(file_name, a.shape, a.max())
#     for i in range(a.shape[0]):
#         img = a[i]
for type in types:
    print(type)
    os.mkdir(os.path.join(dist_path, type))
    os.mkdir(os.path.join(dist_path, type, 'training_input'))
    os.mkdir(os.path.join(dist_path, type, 'training_gt'))
    os.mkdir(os.path.join(dist_path, type, 'validate_input'))
    os.mkdir(os.path.join(dist_path, type, 'validate_gt'))
    low_file = os.path.join(path, type+'-lowsnr.tif')
    high_file = os.path.join(path, type+'-highsnr.tif')
    low = imread(low_file)
    high = imread(high_file)
    inds = np.arange(low.shape[0])
    np.random.shuffle(inds)
    img_train = low[inds[:int(low.shape[0]* 0.9) ]]
    img_valid = low[inds[int(low.shape[0]* 0.9) :]]
    gt_train = high[inds[:int(low.shape[0]* 0.9)]]
    gt_valid = high[inds[int(low.shape[0]* 0.9) :]]
    for i in range(img_train.shape[0]):
        img = img_train[i]
        gt = gt_train[i]
        imwrite(os.path.join(dist_path, type, 'training_input', str(i)+'.tif'), img)
        imwrite(os.path.join(dist_path, type, 'training_gt', str(i) + '.tif'), gt)
    for i in range(img_valid.shape[0]):
        img = img_valid[i]
        gt = gt_valid[i]
        imwrite(os.path.join(dist_path, type, 'validate_input', str(i) + '.tif'), img)
        imwrite(os.path.join(dist_path, type, 'validate_gt', str(i) + '.tif'), gt)