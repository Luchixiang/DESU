import os
import shutil
from tifffile import imread, imwrite
# path = '/mnt/sdb/cxlu/SR_Data/5/'
# # target_path = '/mnt/sdb/cxlu/SR_Data/5_processed/mitochondria'
path = '/userhome/34/cxlu/5/Dataset_SR'
target_path = '/userhome/34/cxlu/5_processed/mitochondria'
if not os.path.exists(os.path.join(target_path, 'training_input')):
    os.mkdir(os.path.join(target_path, 'training_input'))
    os.mkdir(os.path.join(target_path, 'training_gt'))
    os.mkdir(os.path.join(target_path, 'validate_input'))
    os.mkdir(os.path.join(target_path, 'validate_gt'))
FOV_list = ['FOV'+str(d) for d in range(1, 17)]
FOV_train_list = FOV_list[: 13]
FOV_val_list = FOV_list[13:]
print(FOV_list)
# for subdir in os.listdir(path):
# #     if subdir != 'Target':
# #         if subdir in FOV_train_list:
# #             FOV_path = os.path.join(path, subdir)
# #             for file in FOV_path:
# #                 shutil.copy(os.path.join(FOV_path, file), os.path.join(target_path, 'training_input'))
# #                 shutil.copy(os.path.join(path, 'Target', 'W800_P200_6mW_Ax1_'+'I_t1_SRRF.tif')
Flag = False

for i in range(1, 14):
    FOV_path = os.path.join(path, 'FOV'+str(i))
    print(FOV_path)
    for file in os.listdir(FOV_path):
        # print(file)
        print(os.path.join(FOV_path, file), os.path.join(target_path, 'training_input', file))
        shutil.copy(os.path.join(FOV_path, file), os.path.join(target_path, 'training_input', file))
        if not Flag:
            a = imread(os.path.join(FOV_path, file))
            b = imread(os.path.join(path, 'Target', 'W800_P200_6mW_Ax1_FOV_'+str(i).zfill(2)+'_I_t1_SRRF.tif'))
            print(a.shape, b.shape)
            Flag = True
        shutil.copy(os.path.join(path, 'Target', 'W800_P200_6mW_Ax1_FOV_'+str(i).zfill(2)+'_I_t1_SRRF.tif'), os.path.join(target_path, 'training_gt', file))
for i in range(14, 17):
    FOV_path = os.path.join(path, 'FOV'+str(i))
    for file in os.listdir(FOV_path):
        shutil.copy(os.path.join(FOV_path, file), os.path.join(target_path, 'validate_input', file))
        shutil.copy(os.path.join(path, 'Target', 'W800_P200_6mW_Ax1_FOV_'+str(i).zfill(2)+'_I_t1_SRRF.tif'), os.path.join(target_path, 'validate_gt', file))
