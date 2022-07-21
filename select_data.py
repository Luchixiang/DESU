import os
import torch
from model import Expert
import argparse
import numpy as np
import random
from utils import generate_crop
from moe_dataset import SelectDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle


def select_data(args):
    img_list = []
    for file in os.listdir(os.path.join(args.data, 'training_input')):
        img_list.append(os.path.join(args.data, 'training_input', file))
    img_list = generate_crop(img_list, patch_size=512, stride=256)
    transform_list = [transforms.Resize(224)]
    transform_list = transforms.Compose(transform_list)
    dataset = SelectDataset(img_list, transform_list)
    data_loader = DataLoader(dataset, batch_size=64, num_workers=6, shuffle=False)
    out_dict = dict()
    for weight_file in os.listdir(args.weight):
        # print(f'evaluating {weight_file}')
        model = Expert().cuda()
        weight = torch.load(os.path.join(args.weight, weight_file))
        state_dict = weight['state_dict']
        model.load_state_dict(state_dict)
        correct = 0
        model.eval()
        with torch.no_grad():
            for idx, (img, target) in enumerate(data_loader):
                img = img.float().cuda()
                # print(img.shape)
                target = target.long().cuda()
                for i in range(4):
                    img_ = img[:,i, :, :].unsqueeze_(dim=1)
                    target_ = target[:, i]
                    out = model(img_)
                    confidence, predicted = out.max(1)
                    correct += predicted.eq(target_).sum().item()
                    # print(f'rotate num{i}, correct num{predicted.eq(target_).sum().item()}')
            acc = correct / (len(dataset) * 4)
            out_dict[weight_file] = acc
            print(f'evaluate {weight_file} finished and the acc is {acc}')
    with open('transfer_select_16.pkl', 'wb') as f:
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
