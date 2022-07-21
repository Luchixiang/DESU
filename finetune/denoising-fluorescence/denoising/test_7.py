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

from utils.misc import mkdirs, stitch_pathes, to_numpy, module_size
from utils.plot import save_samples, save_stats, plot_row
from utils.metrics import cal_psnr, cal_psnr2, cal_ssim
from utils.data_loader2 import get_test_loader
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
plt.switch_backend('agg')


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='dncnn_encoder', choices=['dncnn', 'n2n'], type=str, help='the model name')
parser.add_argument('-bs', '--batch-size', default=1, type=int, help='test batch size')
# parser.add_argument('--data-root', default='./dataset', type=str, help='dir to dataset')
parser.add_argument('--data-root', default='/mnt/sdb/cxlu/SR_Data/10/fmdd', type=str, help='dir to dataset')
parser.add_argument('--pretrain-dir', default='./experiments/resunet50/Feb_24/dncnn_noise_train[1, 2, 4, 8, 16]_test[1]_captures50_four_crop_epochs400_bs8_lr0.0001', type=str, help='dir to pre-trained model')
parser.add_argument('--noise-levels', default=[1, 2, 4, 8, 16], type=str, help='dir to pre-trained model')
parser.add_argument('--image-types', default=None, type=str, help='image type')
parser.add_argument('--no-cuda', action='store_true', default=False, help='use GPU or not, default using GPU')
args_test = parser.parse_args()

os.environ['CUDA_VISIBILE_DEVICES'] = '0'
device = 'cuda'
#run_dir = args_test.pretrain_dir + f'/{args_test.model}'
run_dir = args_test.pretrain_dir

with open(run_dir + '/args.txt') as args_file:
    args = Namespace(**json.load(args_file))
pprint(args)
if args_test.no_cuda:
    test_dir = run_dir + '/benchmark_cpu'
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
#model = ResUnet()
upsampler = UpsamplerModel(scale=1, n_feat=64)
if args.debug:
    print(model)
    print(module_size(model))
# model.load_state_dict(torch.load(run_dir + f'/checkpoints/model_epoch{args.epochs}.pth',
model.load_state_dict(torch.load(run_dir + f'/checkpoints/model_epoch{400}.pth',
    map_location='cpu'))
#upsampler.load_state_dict(torch.load(run_dir + f'/checkpoints/upsampler_epoch{args.epochs}.pth',
upsampler.load_state_dict(torch.load(run_dir + f'/checkpoints/upsampler_epoch{400}.pth',
    map_location='cpu'))
model = model.to(device)
upsampler.to(device)
model.eval()
upsampler.eval()

logger = {}
# (tl, tr, bl, br, center) --> only select the first four

test_loader = get_test_loader(args.data_root)
psnr, psnr2, ssim, time_taken = 0., 0., 0., 0
for batch_idx, (noisy, clean) in enumerate(test_loader):
    noisy, clean = noisy.to(device), clean.to(device)
    # fuse batch and four crop
    tic_i = time.time()
    denoised = model(noisy)
    denoised = upsampler(denoised)
    time_taken += (time.time() - tic_i)
    psnr += cal_psnr(clean, denoised).sum().item()
    ssim += cal_ssim(clean, denoised).sum()

# time per 512x512 (training image is 256x256)

psnr = psnr / 2
ssim = ssim / 2

result = {'psnr': psnr,
          'ssim': ssim,
          'time': time_taken}
# case.update(result)
pprint(result)
# logger.update({f'noise{noise_level}_{image_type}': case})

# fixed test: (n_plots, 4, 1, 256, 256)
# for i in range(n_plots):
#     print(f'plot {i}-th denoising: [noisy, denoised, clean]')
#     fixed_denoised = model(fixed_test_noisy[i])
#
#     fixed_noisy_stitched = stitch_pathes(to_numpy(fixed_test_noisy[i]))
#     fixed_denoised_stitched = stitch_pathes(to_numpy(fixed_denoised))
#     fixed_clean_stitched = stitch_pathes(to_numpy(fixed_test_clean[i]))
#     plot_row(np.concatenate((fixed_noisy_stitched, fixed_denoised_stitched,
#         fixed_clean_stitched)), test_case_dir, f'denoising{i}',
#         same_range=True, plot_fn='imshow', cmap=cmap, colorbar=False)

with open(test_dir + "/results_{}.txt".format('cpu' if args_test.no_cuda else 'gpu'), 'w') as args_file:
    json.dump(logger, args_file, indent=4)
# print(f'done test in {time.time()-tic} seconds')

# print(f'Finally done in {time.time()-gtic} sec')
