import os
import torch
from model import Expert,UNet, AutoEncoder
import argparse
import numpy as np
import random
from utils import generate_crop
from moe_dataset import SelectDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
from moe_dataset import build_moe_dataset, moe_au_dataset


def select_data(args):
    # img_list = []
    # for file in os.listdir(os.path.join(args.data, 'training_input')):
    #     img_list.append(os.path.join(args.data, 'training_input', file))
    data_loader = moe_au_dataset(args, args.data)
    # img_list = generate_crop(img_list, patch_size=512, stride=256)
    # transform_list = [transforms.Resize(224)]
    # transform_list = transforms.Compose(transform_list)
    # dataset = SelectDataset(img_list, transform_list)
    # data_loader = DataLoader(dataset, batch_size=64, num_workers=6, shuffle=False)
    out_dict = dict()
    criterion = nn.MSELoss().cuda()
    for weight_file in os.listdir(args.weight):
        # print(f'evaluating {weight_file}')
        model = AutoEncoder(num_classes=1).cuda()
        weight = torch.load(os.path.join(args.weight, weight_file))
        state_dict = weight['state_dict']
        model.load_state_dict(state_dict)
        # correct = 0
        loss = 0.
        model.eval()
        with torch.no_grad():
            for idx, (img, target) in enumerate(data_loader):
                # print(img.shape, target.shape)
                img = img.float().cuda()
                # print(img.shape)
                target = target.float().cuda()
                out = model(img)
                loss += criterion(out, target).item()
                    # print(f'rotate num{i}, correct num{predicted.eq(target_).sum().item()}')
            #acc = correct / (len(dataset) * 4)
            loss = loss / len(data_loader)
            out_dict[weight_file] = loss
            print(f'evaluate {weight_file} finished and the acc is {loss}')
    with open('transfer_select_au_16_2.pkl', 'wb') as f:
        pickle.dump(out_dict, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Training benchmark')
    parser.add_argument('--data', metavar='DIR', default='/data1/luchixiang/LUNA16/processed',
                        help='path to dataset')

    parser.add_argument('--b', default=16, type=int, help='batch size')
    parser.add_argument('--weight', default=None, type=str, help='weight to load')
    parser.add_argument('--epoch', default=100, type=int, help='epochs to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--output', default='./model_genesis_pretrain', type=str, help='output path')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='gpu indexs')
    parser.add_argument('--patience', default=50, type=int, help='patience')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--workers', default=6)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    seed = args.seed
    # seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(args)
    select_data(args)
