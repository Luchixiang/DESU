import warnings
import argparse
import os
from model import UNet, UpsamplerModel, ResUnet
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import numpy as np
import random
from train2 import train_cnn
from train_point import train_cnn_point
from dataset_point import BioDataset
from torch.utils.data import DataLoader
import shutil
from collections import OrderedDict
from utils.util import tensor2label, tensor2im
from utils.visualizer import Visualizer
from utils import html


warnings.filterwarnings('ignore')


def train_distributed(args):
    # args.rank = int(os.environ["RANK"])

    model = UNet(num_classes=1, in_channels=1, depth=3).cuda()
    n_feat = 64
    param_dict = []
    upsampler = UpsamplerModel(scale=4, n_feat=n_feat).cuda()
    base_dir = './weight_point'
    weight_path = './weight_point/best.pt'
    weight = torch.load(weight_path)
    state_dict = weight['state_dict']
    unParalled_decoder_state_dict = {}
    # print(net_dict.keys())
    for key in state_dict.keys():
        # if 'down' in key:
        unParalled_decoder_state_dict[key.replace("module.", "")] = state_dict[key]
    model.load_state_dict(unParalled_decoder_state_dict)
    upsampler_dict = weight['upsampler']
    unParalled_upsampler_state_dict = {}
    for key in upsampler_dict.keys():
        unParalled_upsampler_state_dict[key.replace("module.", "")] = upsampler_dict[key]
    upsampler.load_state_dict(unParalled_upsampler_state_dict)
    criterion = torch.nn.MSELoss().cuda()
    dataset_val = BioDataset()
    dataset_val.initialize(False)
    val_loader = DataLoader(dataset=dataset_val, num_workers=args.num_worker, batch_size=4, pin_memory=True,
                            )
    visualizer = Visualizer(args)
    web_dir = os.path.join(base_dir, 'test')
    webpage = html.HTML(web_dir, 'Experiment = a, Phase = b, Epoch = best')
    model = nn.parallel.DataParallel(model)
    upsampler = nn.parallel.DataParallel(upsampler)
    with torch.no_grad():
        model.eval()

        upsampler.eval()
        print("validating....")
        for i, (image, gt, path) in enumerate(val_loader):
            if i == 100:
                break
            loss = 0.
            print(gt.max(), gt.min(), image.max(), image.min())
            # image = image.cuda(non_blocking=True).float()

            image_scale = image.cuda(non_blocking=True).float()
            gt_scale = gt.cuda(non_blocking=True).float()
            # (image_scale, gt_scale) = lazy(image_scale, gt_scale)
            pred = model(image_scale)
            # print(pred.shape)
            pred = upsampler(pred)
            loss += criterion(pred, gt_scale)
            print(loss)
            visuals = OrderedDict([('input_label', tensor2label(image[0], 0)),
                                   ('synthesized_image', tensor2im(pred[0])),
                                   ('real_image', tensor2im(gt[0]))])
            visualizer.save_images(webpage, visuals, path)
    webpage.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Training benchmark')
    parser.add_argument('--data', metavar='DIR', default='/data1/luchixiang/LUNA16/processed',
                        help='path to dataset')
    parser.add_argument('--b', default=16, type=int, help='batch size')
    parser.add_argument('--weight', default=None, type=str, help='weight to load')
    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--epoch', default=100, type=int, help='epochs to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--output', default='./model_genesis_pretrain', type=str, help='output path')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='gpu indexs')
    parser.add_argument('--patience', default=50, type=int, help='patience')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_worker', type=int, default=6)
    parser.add_argument('--warm_up', type=int, default=5)
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--finetune', default='15_')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--pickle_file', default=None)
    parser.add_argument('--sample_num', default=15000)
    parser.add_argument('--uniform', action='store_true', default=False)
    parser.add_argument('--au', action='store_true', default=False)
    parser.add_argument('--pickle_file_mid', default='./select_rotNet_16_mid.pkl')
    parser.add_argument('--similar', action='store_true', default=False)
    parser.add_argument('--simtissue', type=str, default='actin')
    parser.add_argument('--zero', action='store_true', default=False)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    # args.rank = int(os.environ["RANK"])
    # set up the seed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier()

    seed = args.seed
    # seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(args)
    # shutil.copy('./main3.py')
    train_distributed(args)
