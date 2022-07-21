import os
from tifffile import imread, imwrite
import numpy as np
path = '/mnt/sdb/cxlu/SR_Data/15/Mouse_skull/'
dist_path = '/mnt/sdb/cxlu/SR_Data_processed/15_processed/Mouse_skull/'
observation= imread(path+'example2_digital_offset300.tif')
os.mkdir(dist_path)
os.mkdir(os.path.join(dist_path, 'training_input'))
os.mkdir(os.path.join(dist_path, 'training_gt'))
os.mkdir(os.path.join(dist_path, 'validate_input'))
os.mkdir(os.path.join(dist_path, 'validate_gt'))

# The data contains 100 images of a static sample.
# We estimate the clean signal by averaging all images.
signal=np.mean(observation[:,...],axis=0)[np.newaxis, ...]
print(observation.shape, signal.shape, observation[0].max())
inds = np.arange(observation.shape[0])
np.random.shuffle(inds)
x_train = observation[inds[:90]]
x_val = observation[inds[90:]]
for i in range(x_train.shape[0]):
    img = x_train[i]
    gt = signal[0]
    imwrite(os.path.join(dist_path, 'training_input', str(i) + '.tif'), img)
    imwrite(os.path.join(dist_path, 'training_gt', str(i) + '.tif'), gt)
for i in range(x_val.shape[0]):
    img = x_val[i]
    gt = signal[0]
    imwrite(os.path.join(dist_path, 'validate_input', str(i) + '.tif'), img)
    imwrite(os.path.join(dist_path, 'validate_gt', str(i) + '.tif'), gt)

