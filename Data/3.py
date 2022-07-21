import numpy as np
import imageio
from tifffile import imread, imwrite
import os

path = '/mnt/sdb/cxlu/SR_Data/3/Denoising_Planaria/train_data/data_label.npz'
target_path = '/mnt/sdb/cxlu/SR_Data/3_processed/Denoising_Planaria/'
if not os.path.exists(os.path.join(target_path, 'training_input')):
    os.mkdir(os.path.join(target_path, 'training_input'))
    os.mkdir(os.path.join(target_path, 'training_gt'))
    os.mkdir(os.path.join(target_path, 'validate_input'))
    os.mkdir(os.path.join(target_path, 'validate_gt'))
a = np.load(path)
print(a.files)
# print(a['X'].shape)
X, Y = a['X'], a['Y']
print(X.shape)
#print(a['Y'].shape)
# path2 = '/Users/luchixiang/Downloads/disk/SR_Data/3/Denoising_Planaria/test_data/condition_1/EXP278_Smed_fixed_RedDot1_sub_5_N7_m0001.tif'
# b = imread(path2)
# print(b.shape)
train_X = X[:int(X.shape[0] * 0.9)]
train_Y = Y[:int(X.shape[0] * 0.9)]
valid_X = X[int(X.shape[0] * 0.9):]
valid_Y = Y[int(X.shape[0] * 0.9):]
idx = 0

for i in range(train_X.shape[0]):
    img = train_X[i][0]
    gt = train_Y[i][0]
    # print(img.shape)
    imwrite(os.path.join(target_path, 'training_input', str(idx) + '.tif'), img)
    imwrite(os.path.join(target_path, 'training_gt', str(idx) + '.tif'), gt)
    # print(os.path.join(target_path, 'training_input', str(idx) + '.tif'))
    print(idx)
    idx += 1
idx = 0
for i in range(valid_X.shape[0]):
    img = valid_X[i][0]
    gt = valid_Y[i][0]
    imwrite(os.path.join(target_path, 'validate_input', str(idx) + '.tif'), img)
    imwrite(os.path.join(target_path, 'validate_gt', str(idx) + '.tif'), gt)
    #print(os.path.join(target_path, 'validate_input', str(idx) + '.tif'))
    print(idx)
    idx += 1

# path = '/Volumes/TOSHIBA EXT/SR_Data/3/Denoising_Tribolium/train_data/data_label.npz'