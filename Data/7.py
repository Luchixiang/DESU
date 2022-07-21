import os
from tifffile import imread
path = '/mnt/sdb/cxlu/SR_Data/7/CARE3D/actin3d/training_gt'
a = imread(os.path.join(path, '01.tif'))

print(a.shape, a.max())