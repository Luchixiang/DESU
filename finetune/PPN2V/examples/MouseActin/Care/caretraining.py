import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import random

sys.path.append('../../../')
from unet.model import UNet, UpsamplerModel, ResUnet
from pn2v import utils
from pn2v import histNoiseModel
from pn2v import training
from tifffile import imread
import os
import segmentation_models_pytorch as smp
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# See if we can use a GPU
device = utils.getDevice()
# seed = 42
# # seed = args.seed
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
path = '/mnt/sdb/cxlu/SR_Data/15/Mouse_actin/'
# path = '/userhome/34/cxlu/15/Mouse_actin/'
fileName = 'sample_attempt2.tif'
dataName = 'convallaria'  # This will be used to name the "care" model

data = imread(path + fileName)
nameModel = dataName + '_care'

dataGT = np.mean(data, axis=0)[np.newaxis, ..., np.newaxis]
data = data[..., np.newaxis]
dataGT = np.repeat(dataGT, 100, axis=0)
# data = data[:10]
# dataGT = dataGT[:10]
print("Shape of Raw Noisy Image is ", data.shape, "; Shape of Target Image is ", dataGT.shape)
data = np.concatenate((data, dataGT), axis=-1)
print("Shape of `data` is ", data.shape)
# data=np.concatenate( (data[:,512:,512:,:], data[:,:512,512:,:], data[:,512:,:512,:])  )
data = data[:, :,512:, :]
net = UNet(1, depth=3)
#net = ResUnet()
#net = smp.Unet('resnet18', in_channels=1, classes=1)
upsampler = UpsamplerModel(scale=1, n_feat=64)
weight = '/home/cxlu/desu/./weight_1e-1_l3_15/best.pt'
# weight = '/userhome/34/cxlu/desu/mmd/best_actin.pt'
weight = torch.load(weight)
state_dict = weight['state_dict']
unParalled_decoder_state_dict = {}
net_dict = net.state_dict()
for key in state_dict.keys():
    unParalled_decoder_state_dict[key.replace("module.", "")] = state_dict[key]
net_dict.update(unParalled_decoder_state_dict)
net.load_state_dict(net_dict)
upsampler_dict = weight['upsamplers'][0]
unParalled_upsampler_state_dict = {}
for key in upsampler_dict.keys():
    unParalled_upsampler_state_dict[key.replace("module.", "")] = upsampler_dict[key]
# upsampler.load_state_dict(unParalled_upsampler_state_dict)
# Split training and validation data.
my_train_data = data[:-5].copy()
my_val_data = data[-5:].copy()
# for param in net.parameters():
#         param.requires_grad = False
# Start training.
trainHist, valHist = training.trainNetwork(net=net, trainData=my_train_data, valData=my_val_data,
                                           postfix=nameModel, directory='./weight_scratch2', noiseModel=None,
                                           device=device, numOfEpochs=200, stepsPerEpoch=5,
                                           virtualBatchSize=20, batchSize=1, learningRate=1e-3, supervised=True, upsampler=upsampler, freeze=True)

