import os
path = '/Users/luchixiang/Downloads/disk/SR_Data/1/ER'
import shutil
for sub_dir in os.listdir(path):
    gt_file = os.path.join(path, sub_dir, 'GTSIM', 'GTSIM_level_06.mrc')
    if not os.path.exists(gt_file):
        continue
    gt_file_dist = os.path.join(path, sub_dir, 'GTSIM_level_06.mrc')
    raw_path = os.path.join(path, sub_dir, 'RawSIMData')
    raw_path_dist = os.path.join(path, sub_dir)
    shutil.move(gt_file, gt_file_dist)
    for raw_file in os.listdir(raw_path):
        raw_file_src = os.path.join(raw_path, raw_file)
        raw_file_dist = os.path.join(raw_path_dist, raw_file)
        shutil.move(raw_file_src, raw_file_dist)