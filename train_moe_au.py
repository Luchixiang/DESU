import torch
import argparse
import os
import numpy as np
import random
from moe_dataset import build_moe_dataset, moe_au_dataset
import torchvision.models
from model import Expert, UNet,AutoEncoder
from apex import amp
import torch.nn as nn
from timm.scheduler.cosine_lr import CosineLRScheduler
import sys
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def train_moe(args):
    train_losses = []
    for subdir in os.listdir(args.data):
        for tissue in os.listdir(os.path.join(args.data, subdir)):
            data_loader = moe_au_dataset(args, os.path.join(args.data, subdir, tissue))
            num_sample = len(data_loader.dataset)
            # num_epoch = args.epoch * num_sample / 1000
            num_epoch = args.epoch
            model = AutoEncoder(num_classes=1).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            scheduler = CosineLRScheduler(optimizer, t_initial=num_sample * args.epoch,
                                          lr_min=0,
                                          cycle_limit=1,
                                          t_in_epochs=False)
            criterion = nn.MSELoss().cuda()
            print(f'training on {os.path.join(args.data, subdir, tissue)} for {num_epoch} epochs')
            for epoch in range(num_epoch + 1):
                # correct = 0
                train_loss = 0.0
                for idx, (img, target) in enumerate(data_loader):
                    img = img.float().cuda()
                    target = target.float().cuda()
                    # print(img.shape)
                    out = model(img)
                    loss = criterion(out, target)
                   #  print(out.shape, target.shape)
                    optimizer.zero_grad()
                    # confidence, predicted = out.max(1)
                    # correct += predicted.eq(target).sum().item()
                    # acc = utils.compute_acc(output, label)
                    # loss.backward()
                    train_loss += loss.item()
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()
                    scheduler.step_update(epoch * num_sample + idx)
                    train_losses.append(round(loss.item(), 2))
                    if (idx + 1) % 5 == 0:
                        print('Epoch [{}/{}], iteration {}, Loss:{:.6f}, {:.6f} ,learning rate{:.6f}'
                              .format(epoch + 1, args.epoch, idx + 1, loss.item(), np.average(train_losses),
                                      optimizer.state_dict()['param_groups'][0]['lr']))
                        # print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
                        sys.stdout.flush()

                print(f'epoch{epoch} loss {train_loss / len(data_loader)}')
            torch.save({
                'args': args,
                'epoch': num_epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.output, f'expert_{subdir}_{tissue}_last_loss{train_loss / len(data_loader)}.pt'))
            print("Saving model ",
                  os.path.join(args.output, f'expert_{subdir}_{tissue}_last_loss{train_loss / len(data_loader)}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Training benchmark')
    parser.add_argument('--data', metavar='DIR', default='/data1/luchixiang/LUNA16/processed',
                        help='path to dataset')
    parser.add_argument('--b', default=64, type=int, help='batch size')
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
    train_moe(args)
