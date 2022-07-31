"""Benchmark blind Poisson denoising with pretrained models.
Reproduce Table 2 in the paper.

    noise_levels = [1, 2, 4, 8, 16]
    image_types:
        - test_mix
        - type/group_19: 12 types

    metrics:
        - psnr
        - ssim
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.dncnn import DnCNN
from models.unet import UnetN2N
from utils.misc import mkdirs, stitch_pathes, to_numpy, module_size
from utils.plot import save_samples, save_stats, plot_row
from utils.metrics import cal_psnr, cal_psnr2, cal_ssim
from utils.data_loader import load_denoising, load_denoising_test_mix, fluore_to_tensor
import numpy as np
import argparse
from argparse import Namespace
import json
import random
import time
import sys
from pprint import pprint
import matplotlib.pyplot as plt
from unet import UNet, UpsamplerModel, ResUnet
import os


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = image_numpy.astype(imtype)

    return image_numpy


from PIL import Image


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='dncnn_encoder', choices=['dncnn', 'n2n'], type=str, help='the model name')
parser.add_argument('-bs', '--batch-size', default=1, type=int, help='test batch size')
# parser.add_argument('--data-root', default='./dataset', type=str, help='dir to dataset')
parser.add_argument('--data-root', default='/mnt/sdb/cxlu/SR_Data/10/fmdd', type=str, help='dir to dataset')
parser.add_argument('--pretrain-dir',
                    default='./experiment_dissimilar/dncnn_np/Jul_22/dncnn_noise_train[1, 2, 4, 8, 16]_test[1]_captures50_four_crop_epochs400_bs8_lr0.0001',
                    type=str, help='dir to pre-trained model')
parser.add_argument('--noise-levels', default=[1, 2, 4, 8, 16], type=str, help='dir to pre-trained model')
parser.add_argument('--image-types', default="WideField_BPAE_R", type=str, help='image type')
parser.add_argument('--no-cuda', action='store_true', default=False, help='use GPU or not, default using GPU')
args_test = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
test_batch_size = 1
test_seed = 13
cmap = 'inferno'
device = 'cpu' if args_test.no_cuda else 'cuda'

noise_levels = args_test.noise_levels
if args_test.image_types is not None:
    image_types = args_test.image_types
    image_types = [image_types]
    assert isinstance(image_types, (list, tuple))
else:
    image_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
                   'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
                   'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',
                   'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B',
                   'test_mix']
    # image_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
    #                'TwoPhoton_MICE']
image_types = ['test_mix']
# run_dir = args_test.pretrain_dir + f'/{args_test.model}'
run_dir = args_test.pretrain_dir

with open(run_dir + '/args.txt') as args_file:
    args = Namespace(**json.load(args_file))
pprint(args)
if args_test.no_cuda:
    c = run_dir + '/benchmark_cpu'
else:
    test_dir = run_dir + '/benchmark_gpu'
mkdirs(test_dir)
#
# if args_test.model == 'dncnn':
#     model = DnCNN(depth=args.depth,
#                 n_channels=args.width,
#                 image_channels=1,
#                 use_bnorm=True,
#                 kernel_size=3)
# elif args_test.model == 'n2n':
#     model = UnetN2N(args.in_channels, args.out_channels)
model = UNet(1, depth=3)

# model = ResUnet()
upsampler = UpsamplerModel(scale=1, n_feat=64)
if args.debug:
    print(model)
    print(module_size(model))
# model.load_state_dict(torch.load(run_dir + f'/checkpoints/model_epoch{args.epochs}.pth',
model.load_state_dict(torch.load(run_dir + f'/checkpoints/model_epoch{300}.pth',
                                 map_location='cpu'))
# upsampler.load_state_dict(torch.load(run_dir + f'/checkpoints/upsampler_epoch{args.epochs}.pth',
upsampler.load_state_dict(torch.load(run_dir + f'/checkpoints/upsampler_epoch{300}.pth',
                                     map_location='cpu'))
model = model.to(device)
upsampler.to(device)
model.eval()
upsampler.eval()

logger = {}
# (tl, tr, bl, br, center) --> only select the first four
four_crop = transforms.Compose([
    transforms.FiveCrop(args.imsize),
    transforms.Lambda(lambda crops: torch.stack([
        fluore_to_tensor(crop) for crop in crops[:4]])),
    transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
])

gtic = time.time()
record = dict()
for noise_level in noise_levels:
    for image_type in image_types:
        test_case_dir = test_dir + f'/noise{noise_level}_{image_type}'
        mkdirs(test_case_dir)
        tic = time.time()
        if image_type == 'test_mix':
            n_plots = 12
            test_loader = load_denoising_test_mix(args_test.data_root,
                                                  batch_size=test_batch_size, noise_levels=[noise_level],
                                                  transform=four_crop, target_transform=four_crop,
                                                  patch_size=args.imsize)
        else:
            n_plots = 2
            test_loader = load_denoising(args_test.data_root, train=False,
                                         batch_size=test_batch_size, noise_levels=[noise_level],
                                         types=[image_type], captures=50,
                                         transform=four_crop, target_transform=four_crop,
                                         patch_size=args.imsize)

        # four crop
        multiplier = 4
        n_test_samples = len(test_loader.dataset) * multiplier

        np.random.seed(test_seed)
        fixed_idx = np.random.permutation(len(test_loader.dataset))[:n_plots]
        print(f'fixed test index: {fixed_idx}')

        # (n_plots, 4, 1, 256, 256)
        fixed_test_noisy = torch.stack([(test_loader.dataset[i][0]) for i in fixed_idx])
        fixed_test_clean = torch.stack([(test_loader.dataset[i][1]) for i in fixed_idx])
        print(f'fixed test noisy shape: {fixed_test_noisy.shape}')
        fixed_test_noisy = fixed_test_noisy.to(device)

        case = {'noise': noise_level,
                'type': image_type,
                'samples': n_test_samples,
                }
        pprint(case)
        print('Start testing............')

        psnr, psnr2, ssim, time_taken = 0., 0., 0., 0
        for batch_idx, (noisy, clean, noisy_file) in enumerate(test_loader):
            noisy, clean = noisy.to(device), clean.to(device)
           #  print(noisy.shape)

            # print(noisy.shape)
            # fuse batch and four crop
            noisy = noisy.view(-1, *noisy.shape[2:])
            clean = clean.view(-1, *clean.shape[2:])
           #  print(noisy.shape)
            tic_i = time.time()
            denoised = model(noisy)
            denoised = upsampler(denoised)
            time_taken += (time.time() - tic_i)
            psnr_tmp = cal_psnr(clean, denoised).sum().item() / multiplier
            # print(psnr_tmp)
            record['test_' + str(batch_idx) +'_noiselevel' + str(noise_level)+'_' + 'noisy' + '_' + '.png'] = psnr_tmp
            # samples = torch.cat((noisy[:4], denoised[:4], clean[:4], denoised[:4] - clean[:4]))
            # save_samples('./full_200', samples, 0, 'test_' + str(batch_idx) + '_noiselevel' + str(noise_level), epoch=True, cmap='inferno',
            #              heatmap=True)
            name = ['noisy', 'denoised', 'gt']
            # for i in range(4):
            #     image_numpy = tensor2im(noisy[i].unsqueeze_(0))
            #     save_image(image_numpy, os.path.join('./dissimilar_100fake', 'test_' + str(batch_idx) +'_noiselevel' + str(noise_level)+'_' + 'noisy' + '_' + str(i) + '.png'))
            #     image_numpy = tensor2im(denoised[i].unsqueeze_(0))
            #     save_image(image_numpy, os.path.join('./dissimilar_100fake',
            #                                          'test_' + str(batch_idx) + '_noiselevel' + str(noise_level)+'_' + 'denoised' + '_' + str(i) + '.png'))
            #     image_numpy = tensor2im(clean[i].unsqueeze_(0))
            #     save_image(image_numpy, os.path.join('./dissimilar_100fake',
            #                                          'test_' + str(batch_idx) + '_noiselevel' + str(noise_level)+'_' + 'gt' + '_' + str(i) + '.png'))
                # image_numpy = tensor2im((clean[i] - denoised[i]).unsqueeze_(0))
                # save_image(image_numpy, os.path.join('./full_400fake',
                #                                      'test_' + str(batch_idx) + '_noiselevel' + str(noise_level)+'_' + 'gt-denosie' + '_' + str(i) + '.png'))
                # image_numpy = tensor2im((clean[i] - noisy[i]).unsqueeze_(0))
                # save_image(image_numpy, os.path.join('./full_400fake',
                #                                      'test_' + str(batch_idx) + '_noiselevel' + str(noise_level)+'_' + 'gt-noisy' + '_' + str(
                #                                          i) + '.png'))

            psnr += psnr_tmp * multiplier
            ssim += cal_ssim(clean, denoised).sum()

        # time per 512x512 (training image is 256x256)
        time_taken /= (n_test_samples / multiplier)
        psnr = psnr / n_test_samples
        ssim = ssim / n_test_samples

        result = {'psnr': psnr,
                  'ssim': ssim,
                  'time': time_taken}
        case.update(result)
        pprint(result)
        logger.update({f'noise{noise_level}_{image_type}': case})
        with open('dissimilar_200.json', 'w') as fp:
            json.dump(record, fp)
        with open(test_dir + "/results_{}.txt".format('cpu' if args_test.no_cuda else 'gpu'), 'w') as args_file:
            json.dump(logger, args_file, indent=4)
        print(f'done test in {time.time() - tic} seconds')

print(f'Finally done in {time.time() - gtic} sec')
