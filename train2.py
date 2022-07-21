import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torch.distributed as dist
from apex import amp
from timm.scheduler.cosine_lr import CosineLRScheduler
from util import Mixup, mixup_data
from koila import LazyTensor, lazy


def train_cnn(optimizer, model, train_generators, valid_generator, criterion, args, upsamplers):
    # model = nn.DataParallel(model)
    # model.to(device)
    n_iteration_per_epoch = 0
    for i in range(4):
        train_generator = train_generators[i]
        if train_generator != None:
            n_iteration_per_epoch += len(train_generator)
    print(n_iteration_per_epoch)
    scheduler = CosineLRScheduler(optimizer, t_initial=n_iteration_per_epoch * args.epoch,
                                  lr_min=0,
                                  warmup_lr_init=1e-5,
                                  warmup_t=n_iteration_per_epoch * args.warm_up,
                                  cycle_limit=1,
                                  t_in_epochs=False)
    if args.mixup:
        mixuper = Mixup(bs=args.b)
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    best_loss = 100000
    num_epoch_no_improvement = 0
    for epoch in range(args.epoch + 1):
        model.train()
        for upsampler in upsamplers:
            upsampler.train()
        iteration = 0
        for i in range(4):
            train_generator = train_generators[i]
            if train_generator == None:
                continue
            for idx, (image, gt, scale) in enumerate(train_generator):
                loss = 0.
                img = image.cuda(non_blocking=True).float()
                # gt_scale = gt_scale.view(-1, gt_scale[0].shape)
                gt = gt.cuda(non_blocking=True).float()
                # if args.mixup and image_scale.shape[0] > 1:
                #     img, gt = mixuper(img, gt)
                # (image_scale, gt_scale) = lazy(image_scale, gt_scale)
                # scale = scale.cuda().float()
                pred = model(img)
                pred = upsamplers[i](pred)
                # print(pred.shape, gt.shape)
                loss += criterion(pred, gt)
                iteration += 1
                optimizer.zero_grad()
                # loss.backward()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                scheduler.step_update(epoch * n_iteration_per_epoch + iteration)
                train_losses.append(round(loss.item(), 2))

                if (iteration + 1) % 20 == 0:
                    print('Epoch [{}/{}], iteration {}, Loss:{:.6f}, {:.6f} ,learning rate{:.6f}'
                          .format(epoch + 1, args.epoch, iteration + 1, loss.item(), np.average(train_losses),
                                  optimizer.state_dict()['param_groups'][0]['lr']))
                    # print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
                    sys.stdout.flush()
        with torch.no_grad():
            model.eval()
            for upsampler in upsamplers:
                upsampler.eval()
            print("validating....")
            for i, (image, gt, scale) in enumerate(valid_generator):
                loss = 0.
                # image = image.cuda(non_blocking=True).float()
                for idx_scale in [0, 1, 2, 3]:
                    # index = (scale == idx_scale)
                    image_scale = image[idx_scale]
                    gt_scale = gt[idx_scale]
                    if image_scale is None:
                        continue
                    image_scale = image_scale.cuda(non_blocking=True).float()
                    gt_scale = gt_scale.cuda(non_blocking=True).float()
                    # (image_scale, gt_scale) = lazy(image_scale, gt_scale)
                    pred = model(image_scale)
                    # print(pred.shape)
                    pred = upsamplers[idx_scale](pred)
                    loss += criterion(pred, gt_scale)
                valid_losses.append(loss.item())
        # logging
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_loss,
                                                                                    train_loss))
        train_losses = []
        valid_losses = []

        if valid_loss < best_loss:
            print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            # save model
            # save all the weight for 3d unet
            torch.save({
                'args': args,
                'epoch': epoch + 1,
                'upsamplers': [upsampler.state_dict() for upsampler in upsamplers],
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.output,
                            'best' + '.pt'))
            print("Saving model ",
                  os.path.join(args.output,
                               'best' + '.pt'))
        else:
            if epoch % 20 == 0:
                torch.save({
                    'args': args,
                    'epoch': epoch + 1,
                    'upsamplers': [upsampler.state_dict() for upsampler in upsamplers],
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(args.output,
                                'epoch_' + str(epoch) + '.pt'))
                print("Saving model ",
                      os.path.join(args.output,
                                   'epoch_' + str(epoch) + '.pt'))
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                      num_epoch_no_improvement))
            num_epoch_no_improvement += 1
            if num_epoch_no_improvement == args.patience:
                print("Ea`rly Stopping")
                break
        sys.stdout.flush()
