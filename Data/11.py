import os
import shutil

raw_path = '/mnt/sdc/cxlu/11'
dist_path = '/mnt/sdb/cxlu/SR_Data_processed/11'
for tissue in os.listdir(raw_path):
    if tissue == 'actin':
        dir_src = 'actin/train/ac-15/t0'
        dir_tgt = 'actin/train/phalloidin/t0'
    elif tissue == 'tubulin':
        dir_src = 'tubulin/train/dm1a/t0'
        dir_tgt = 'tubulin/train/yol1-34/t0'
    elif tissue == 'paxillin':
        dir_src = 'paxillin/train/5h11/t0'
        dir_tgt = 'paxillin/train/y113/t0'
    elif tissue == 'caveolae':
        dir_src = 'caveolae/train/d1p6w/t0'
        dir_tgt = 'caveolae/train/4h312/t0'
    os.mkdir(os.path.join(dist_path, tissue))
    os.mkdir(os.path.join(dist_path, tissue, 'training_input'))
    os.mkdir(os.path.join(dist_path, tissue, 'training_gt'))
    os.mkdir(os.path.join(dist_path, tissue, 'validate_gt'))
    os.mkdir(os.path.join(dist_path, tissue, 'validate_input'))
    for file in os.listdir(os.path.join(raw_path, dir_src)):
        shutil.copy(os.path.join(raw_path, dir_src, file), os.path.join(dist_path, tissue, 'training_input', file))
        shutil.copy(os.path.join(raw_path, dir_tgt, file), os.path.join(dist_path, tissue, 'training_gt', file))
        print(file)



