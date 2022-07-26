import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import *
from utils import *
from unet import UNet, UpsamplerModel
from tifffile import imread
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']= '0'
parser = argparse.ArgumentParser(description="Inference")
parser.add_argument("--epoch", type=int, default=49, help="Number of training epochs")
parser.add_argument("--net", type=str, default="D", help="DnCNN (D), MemNet (M), RIDNet (R)")
parser.add_argument("--img_avg", type=int, default=2, help="Number of images pre-averaged (determines the noise level)")

parser.add_argument("--test_path", type=str, default='/mnt/sdb/cxlu/SR_Data_processed/16_denoise/tissue/validate_input',
                    help="Directory where test images are stored")
parser.add_argument("--gt_path", type=str, default='/mnt/sdb/cxlu/SR_Data_processed/16_denoise/tissue/validate_gt', help="Directory where GT images are stored")
opt = parser.parse_args()


def norm(img, percentage_low, percentage_high):
    # img = (img - np.percentile(img, percentage_low)) / (
    #         np.percentile(img, percentage_high) - np.percentile(img, percentage_low) + 1e-7)
    # img[img > 1] = 1
    # img[img < 0] = 0
    img = (img - 154.5) / 66.028
    return img
def inference(model, upsampler, test_path, gt_path, results_dir):
    # files_noisy = glob.glob(os.path.join(test_path, '*.npy'))[240:]
    # files_noisy.sort()
    # files_gt = glob.glob(os.path.join(gt_path, '*.npy'))[240:]
    # files_gt.sort()
    # print(len(files_noisy))
    files_noisy = []
    files_gt = []
    for image in os.listdir(test_path):
        if int(image.split('_')[1][3:-4]) == opt.img_avg:
            files_noisy.append(os.path.join(test_path, image))
            files_gt.append(os.path.join(gt_path, image))
    print(len(files_noisy))
    psnr_results = np.zeros(len(files_noisy))
    ssim_results = np.zeros(len(files_noisy))

    for idx in range(len(files_noisy)):
        # Img = cv2.imread(files_noisy[idx], cv2.IMREAD_GRAYSCALE)

        # Img = np.load(files_noisy[idx])
        # Img = Img / 255.
        Img = imread(files_noisy[idx])
        Img = norm(Img, 1, 99)
        Img = Img / 255.
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        img_noisy = torch.Tensor(Img)


        img_noisy = Variable(img_noisy.cuda())

         #Img = np.load(files_gt[idx])
        Img = imread(files_gt[idx])
        Img = norm(Img, 1, 99)
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        img_gt = torch.Tensor(Img)
        img_gt = Variable(img_gt.cuda())

        # feed forward then clamp image
        with torch.no_grad():
            if opt.net == 'R':
                noise_prediction = img_noisy - model(img_noisy)
            else:
                noise_prediction = model(img_noisy)
                noise_prediction = upsampler(noise_prediction)
            #             else:
            #                 NoiseNetwork, NoiseLevelNetwork = model(INoisy)

            img_prediction = img_noisy - noise_prediction
            # img_prediction = torch.clamp(img_prediction, 0., 1.)
            img_prediction = img_prediction * 255.

            psnr_results[idx] = batch_PSNR(img_prediction, img_gt, img_gt.max().cpu() - img_gt.min().cpu())
            ssim_results[idx] = batch_SSIM(img_prediction, img_gt, math.ceil(img_gt.max().cpu() - img_gt.min().cpu()))

            # SAVE IMAGES
            # img_prediction = np.squeeze(img_prediction.detach().cpu().numpy())
            # cv2.imwrite(os.path.join(results_dir, os.path.basename(files_noisy[idx])), (img_prediction).astype('uint8'))

    return psnr_results, ssim_results


def main():
    model_name = f'{opt.net}_{opt.img_avg}'
    model_dir = os.path.join('../net_data/trained_denoisers_avg2_npwarm/', model_name)



    print('Testing with model %s at epoch %d, with %s' % (model_name, opt.epoch, opt.test_path))

    # model_channels = 1
    # if opt.net == 'D':
    #     net = DnCNN(channels=model_channels, num_of_layers=17)
    # elif opt.net == 'M':
    #     net = MemNet(in_channels=model_channels)
    # elif opt.net == 'R':
    #     net = RIDNET(in_channels=model_channels)
    # else:
    #     raise NotImplemented('Network model not implemented.')
    model = UNet(1, depth=3).cuda()
    # model = nn.DataParallel(net).cuda()
    upsampler = UpsamplerModel(scale=1).cuda()
    model.load_state_dict(torch.load(os.path.join(model_dir, 'epoch_%d.pth' % (opt.epoch))))
    upsampler.load_state_dict(torch.load(os.path.join(model_dir, 'upsampler_epoch_%d.pth' % (opt.epoch))))
    model.eval()
    upsampler.eval()

    # Run inference
    results_dir = os.path.join('./results/', model_name, os.path.basename(opt.test_path))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    psnr_results, ssim_results = inference(model, upsampler, opt.test_path, opt.gt_path, results_dir)

    print('Average %s PSNR: %.4fdB' % (opt.test_path, np.mean(psnr_results)))
    print('Average %s PSNR: %.4fdB' % (opt.test_path, np.mean(ssim_results)))
    np.save(os.path.join(results_dir, 'PSNR'), psnr_results)
    np.save(os.path.join(results_dir, 'SSIM'), ssim_results)


if __name__ == "__main__":
    main()
