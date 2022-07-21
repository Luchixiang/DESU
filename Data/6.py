import os
from tifffile import imread
path = '/mnt/sdb/cxlu/SR_Data/6/Confocal_2_STED/SirDNA/Training/'
gt_path = os.path.join(path, 'GT', 'Processed_Res-to-STED Fixed_9gt.tif')
raw_path = os.path.join(path, 'RAW', 'Processed_Res-to-STED Fixed_9raw.tif')
gt = imread(gt_path)
raw = imread(raw_path)
print(gt.shape)
print(raw.shape)