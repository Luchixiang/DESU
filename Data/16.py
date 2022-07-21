import numpy as np
from tifffile import imwrite
import os
import numpy as np
from PIL import Image

raw_path = '/mnt/sdc/cxlu/16/raw'
hr_dist_path = '/mnt/sdb/cxlu/SR_Data_processed/16_hr'
denoise_dist_path = '/mnt/sdb/cxlu/SR_Data_processed/16_denoise/tissue'
# os.mkdir(hr_dist_path)
# os.mkdir(os.path.join(hr_dist_path, 'training_input'))
# os.mkdir(os.path.join(hr_dist_path, 'training_gt'))
# os.mkdir(os.path.join(hr_dist_path, 'validate_input'))
# os.mkdir(os.path.join(hr_dist_path, 'validate_gt'))
os.mkdir(denoise_dist_path)
os.mkdir(os.path.join(denoise_dist_path, 'training_input'))
os.mkdir(os.path.join(denoise_dist_path, 'training_gt'))
os.mkdir(os.path.join(denoise_dist_path, 'validate_input'))
os.mkdir(os.path.join(denoise_dist_path, 'validate_gt'))

for image in os.listdir(raw_path):
    for i in range(3):
        hr_image =  os.path.join(raw_path, image, f'sim_channel{i}.npy')
        raw_image = os.path.join(raw_path, image, f'wf_channel{i}.npy')
        hr_image = np.load(hr_image)
        raw_image = np.load(raw_image)
        avg1_imge = raw_image[249]
        avg2_imge = np.mean(np.stack([raw_image[0], raw_image[1]], axis=0), axis=0)
        avg4_imge = np.mean(np.stack([raw_image[0], raw_image[1], raw_image[2], raw_image[3]], axis=0), axis=0)
        avg8_imge = np.mean(raw_image[:8], axis=0)
        avg16_imge = np.mean(raw_image[:16], axis=0)
        # avg1_imge = Image.fromarray(avg1_imge).convert("L")
        # avg2_imge = Image.fromarray(avg2_imge).convert("L")
        # avg4_imge = Image.fromarray(avg4_imge).convert("L")
        # avg8_imge = Image.fromarray(avg8_imge).convert("L")
        # avg16_imge = Image.fromarray(avg16_imge).convert("L")
        # gt_imge = Image.fromarray(np.mean(raw_image, axis=0)).convert("L")
        if int(image[-3:]) <= 80:
            # avg1_imge.save(os.path.join(denoise_dist_path, 'training_input', image + f'channel{i}_avg1.png'))
            # avg2_imge.save(os.path.join(denoise_dist_path, 'training_input', image + f'channel{i}_avg2.png'))
            # avg4_imge.save(os.path.join(denoise_dist_path, 'training_input', image + f'channel{i}_avg4.png'))
            # avg8_imge.save(os.path.join(denoise_dist_path, 'training_input', image + f'channel{i}_avg8.png'))
            # avg16_imge.save(os.path.join(denoise_dist_path, 'training_input', image + f'channel{i}_avg16.png'))
            # gt_imge.save(os.path.join(denoise_dist_path, 'training_gt', image + f'channel{i}_avg1.png'))
            # gt_imge.save(os.path.join(denoise_dist_path, 'training_gt', image + f'channel{i}_avg2.png'))
            # gt_imge.save(os.path.join(denoise_dist_path, 'training_gt', image + f'channel{i}_avg4.png'))
            # gt_imge.save(os.path.join(denoise_dist_path, 'training_gt', image + f'channel{i}_avg8.png'))
            # gt_imge.save(os.path.join(denoise_dist_path, 'training_gt', image + f'channel{i}_avg16.png'))

            # imwrite(os.path.join(hr_dist_path, 'training_gt', image + f'channel{i}.png'), hr_image)
            # imwrite(os.path.join(hr_dist_path, 'training_input', image + f'channel{i}.png'), avg1_imge)

            imwrite(os.path.join(denoise_dist_path, 'training_input', image+f'channel{i}_avg1.tif'), avg1_imge)
            imwrite(os.path.join(denoise_dist_path, 'training_input', image+f'channel{i}_avg2.tif'), avg2_imge)
            imwrite(os.path.join(denoise_dist_path, 'training_input', image+f'channel{i}_avg4.tif'), avg4_imge)
            imwrite(os.path.join(denoise_dist_path, 'training_input', image+f'channel{i}_avg8.tif'), avg8_imge)
            imwrite(os.path.join(denoise_dist_path, 'training_input', image+f'channel{i}_avg16.tif'), avg16_imge)
            imwrite(os.path.join(denoise_dist_path, 'training_gt', image + f'channel{i}_avg1.tif'),np.mean(raw_image, axis=0))
            imwrite(os.path.join(denoise_dist_path, 'training_gt', image + f'channel{i}_avg2.tif'),np.mean(raw_image, axis=0))
            imwrite(os.path.join(denoise_dist_path, 'training_gt', image + f'channel{i}_avg4.tif'),np.mean(raw_image, axis=0))
            imwrite(os.path.join(denoise_dist_path, 'training_gt', image + f'channel{i}_avg8.tif'),np.mean(raw_image, axis=0))
            imwrite(os.path.join(denoise_dist_path, 'training_gt', image + f'channel{i}_avg16.tif'),np.mean(raw_image, axis=0))
        else:
            # avg1_imge.save(os.path.join(denoise_dist_path, 'validate_input', image + f'channel{i}_avg1.png'))
            # avg2_imge.save(os.path.join(denoise_dist_path, 'validate_input', image + f'channel{i}_avg2.png'))
            # avg4_imge.save(os.path.join(denoise_dist_path, 'validate_input', image + f'channel{i}_avg4.png'))
            # avg8_imge.save(os.path.join(denoise_dist_path, 'validate_input', image + f'channel{i}_avg8.png'))
            # avg16_imge.save(os.path.join(denoise_dist_path, 'validate_input', image + f'channel{i}_avg16.png'))
            # gt_imge.save(os.path.join(denoise_dist_path, 'validate_gt', image + f'channel{i}_avg1.png'))
            # gt_imge.save(os.path.join(denoise_dist_path, 'validate_gt', image + f'channel{i}_avg2.png'))
            # gt_imge.save(os.path.join(denoise_dist_path, 'validate_gt', image + f'channel{i}_avg4.png'))
            # gt_imge.save(os.path.join(denoise_dist_path, 'validate_gt', image + f'channel{i}_avg8.png'))
            # gt_imge.save(os.path.join(denoise_dist_path, 'validate_gt', image + f'channel{i}_avg16.png'))
            # imwrite(os.path.join(hr_dist_path, 'validate_gt', image + f'channel{i}.tif'), hr_image)
            # imwrite(os.path.join(hr_dist_path, 'validate_input', image + f'channel{i}.tif'), avg1_imge)
            imwrite(os.path.join(denoise_dist_path, 'validate_input', image + f'channel{i}_avg1.tif'), avg1_imge)
            imwrite(os.path.join(denoise_dist_path, 'validate_input', image + f'channel{i}_avg2.tif'), avg2_imge)
            imwrite(os.path.join(denoise_dist_path, 'validate_input', image + f'channel{i}_avg4.tif'), avg4_imge)
            imwrite(os.path.join(denoise_dist_path, 'validate_input', image + f'channel{i}_avg8.tif'), avg8_imge)
            imwrite(os.path.join(denoise_dist_path, 'validate_input', image + f'channel{i}_avg16.tif'), avg16_imge)
            imwrite(os.path.join(denoise_dist_path, 'validate_gt', image + f'channel{i}_avg1.tif'),
                    np.mean(raw_image, axis=0))
            imwrite(os.path.join(denoise_dist_path, 'validate_gt', image + f'channel{i}_avg2.tif'),
                    np.mean(raw_image, axis=0))
            imwrite(os.path.join(denoise_dist_path, 'validate_gt', image + f'channel{i}_avg4.tif'),
                    np.mean(raw_image, axis=0))
            imwrite(os.path.join(denoise_dist_path, 'validate_gt', image + f'channel{i}_avg8.tif'),
                    np.mean(raw_image, axis=0))
            imwrite(os.path.join(denoise_dist_path, 'validate_gt', image + f'channel{i}_avg16.tif'),
                    np.mean(raw_image, axis=0))
    # hr_image_0 = os.path.join(raw_path, image, 'sim_channel0.npy')
    # hr_image_1 = os.path.join(raw_path, image, 'sim_channel1.npy')
    # hr_image_2 = os.path.join(raw_path, image, 'sim_channel2.npy')
    # raw_image_0 = os.path.join()

