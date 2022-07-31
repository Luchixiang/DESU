import os
data_root = '/mnt/sdb/cxlu/SR_Data_processed'
from tifffile import imread
for subdir in os.listdir(data_root):
    subdir_path = os.path.join(data_root, subdir)
    for tissue in os.listdir(subdir_path):
        tissue_path = os.path.join(subdir_path, tissue)
        sample_imgs = list(os.listdir(os.path.join(tissue_path, 'training_input')))[:5]
        prefix = subdir + '_' + tissue + '_'
        for img in sample_imgs:

            gt_img = os.path.join(tissue_path, 'training_gt', img)
            img_path = os.path.join(tissue_path, 'training_input', img)
            import shutil
            shutil.copy(gt_img, os.path.join('./sample_gt', prefix + img))
            shutil.copy(img_path, os.path.join('./sample', prefix + img))
            if img.endswith('tif'):
                img_np = imread(img_path)
                gt_np = imread(gt_img)
                from PIL import Image
                image_np = Image.fromarray(img_np).convert('L')
                image_gt = Image.fromarray(gt_np).convert('L')
                image_np.save(os.path.join('./sample', prefix + img.replace('.tif', '.png')))
                image_gt.save(os.path.join('./sample_gt', prefix +  img.replace('.tif', '.png')))
