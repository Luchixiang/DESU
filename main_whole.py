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
from train import train_cnn
from dataset import supervised_dataloader
import shutil

warnings.filterwarnings('ignore')


def train_distributed(args):
    # args.rank = int(os.environ["RANK"])
    if args.model == 'res':
        model = ResUnet().cuda()
        n_feat = 16
    else:
        model = UNet(num_classes=1, depth=3).cuda()
        n_feat = 64
    upsamplers = []
    param_dict = []
    for i in range(4):
        upsampler = UpsamplerModel(scale=i + 1, n_feat=n_feat).cuda()
        upsamplers.append(upsampler)
        param_dict.append({'params': upsampler.parameters()})
    param_dict.append({'params': model.parameters()})
    optimizer = torch.optim.SGD(param_dict, lr=args.lr)  # todo whether give the upsamplers different learning rate
    model_list, optimizer = amp.initialize(upsamplers + [model], optimizer, opt_level='O1')
    model = model_list[-1]
    upsamplers = model_list[:-1]
    criterion = torch.nn.MSELoss().cuda()
    #criterion = torch.nn.L1Loss().cuda()
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],  broadcast_buffers=False)
    # upsamplers_distributed = [nn.parallel.DistributedDataParallel(upsampler, device_ids=[args.local_rank], find_unused_parameters=True,  broadcast_buffers=False) for upsampler
    #                           in upsamplers]
    model = nn.parallel.DataParallel(model)
    upsamplers = [nn.parallel.DataParallel(upsampler) for upsampler in upsamplers]
    train_loader, val_loader = supervised_dataloader(args)
    train_cnn(train_generator=train_loader, valid_generator=val_loader, args=args, optimizer=optimizer, model=model,
              criterion=criterion, upsamplers=upsamplers)


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
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--nodes', default=0, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23459', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', type=int, required=True)
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--num_worker', type=int, default=6)
    parser.add_argument('--warm_up', type=int, default=5)
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--finetune', default='15_')
    parser.add_argument('--amp', action='store_true', default=False)
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
    # shutil.copy('./main_whole.py')
    train_distributed(args)
