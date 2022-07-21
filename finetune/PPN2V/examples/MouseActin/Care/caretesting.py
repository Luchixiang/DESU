import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../../')
import torch
from tifffile import imread
import os
from unet.model import UNet
from pn2v.utils import denormalize
from pn2v.utils import normalize
from pn2v.utils import PSNR
from pn2v import utils
from pn2v import prediction
import pn2v.training
from pn2v import histNoiseModel

# See if we can use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = utils.getDevice()
path = '/mnt/sdb/cxlu/SR_Data/15/Mouse_actin/'
# path = '/userhome/34/cxlu/15/Mouse_actin/'

# Load the test data
dataTest = imread(path + "sample_attempt2.tif")[:,:,:512]
# We are loading only a sub image to spped up computation

# We estimate the ground truth by averaging.
dataTestGT = np.mean(dataTest[:, ...], axis=0)[np.newaxis, ...]
dataName = 'convallaria'  # Use the same name as used in 1_CareTraining.ipynb. This is a part of model name created in 1_CareTraining.ipynb
nameModel = dataName + '_care'
net = torch.load('./weight_scratch2' + "/best_" + nameModel + ".net")
upsampler = torch.load('./weight_scratch2' + "/upsampler_best_" + nameModel + ".net")


careRes = []
resultImgs = []
inputImgs = []

# We iterate over all test images.
for index in range(dataTest.shape[0]):
    im = dataTest[index]
    gt = dataTestGT[0]  # The ground truth is the same for all images

    # We are using tiling to fit the image into memory
    # If you get an error try a smaller patch size (ps)
    careResult = prediction.tiledPredict(im, net,upsampler, ps=256, overlap=48,
                                         device=device, noiseModel=None, outScaling=10)
    inputImgs.append(im)

    rangePSNR = np.max(gt) - np.min(gt)
    carePrior = PSNR(gt, careResult, rangePSNR)
    careRes.append(carePrior)

    print("image:", index)
    print("PSNR input", PSNR(gt, im, rangePSNR))
    print("PSNR CARE", carePrior)  # Without info from masked pixel
    print('-----------------------------------')
print("Avg PSNR CARE:", np.mean(np.array(careRes)), '+-(2SEM)',
      2 * np.std(np.array(careRes)) / np.sqrt(float(len(careRes))))
