import os
import math
from torch.distributions import Beta
import torch
import numpy as np
from tifffile import imread
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


def get_train_list(root_dir, finetune_data):
    img_list = []
    gt_list = []
    for subdir in os.listdir(root_dir):
        if finetune_data in subdir:
            continue
        subdir_path = os.path.join(root_dir, subdir)
        for tissue in os.listdir(subdir_path):
            tissue_path = os.path.join(subdir_path, tissue)
            for file in os.listdir(os.path.join(tissue_path, 'training_input')):
                img_list.append(os.path.join(tissue_path, 'training_input', file))

                assert os.path.exists(os.path.join(tissue_path, 'training_gt', file)), print(
                    os.path.join(tissue_path, 'training_gt', file))
                gt_list.append(os.path.join(tissue_path, 'training_gt', file))
                if '10_' not in subdir:
                    img_list.append(os.path.join(tissue_path, 'training_input', file))
                    gt_list.append(os.path.join(tissue_path, 'training_gt', file))
    return img_list, gt_list


def get_selected_train_list(root_dir, finetune_data, weight_dict):
    img_list = []
    gt_list = []
    weighted_values = []
    scale_list = []
    for subdir in os.listdir(root_dir):
        if finetune_data in subdir:
            continue
        subdir_path = os.path.join(root_dir, subdir)
        for tissue in os.listdir(subdir_path):
            tissue_path = os.path.join(subdir_path, tissue)
            sample_img = os.path.join(tissue_path, 'training_input',
                                      os.listdir(os.path.join(tissue_path, 'training_input'))[0])
            sample_gt = os.path.join(tissue_path, 'training_gt', os.path.basename(sample_img))
            if sample_img.endswith('.tif'):
                sample_img = imread(sample_img).astype(np.float)
                sample_gt = imread(sample_gt).astype(np.float)

            elif sample_img.endswith('.png'):
                sample_img = np.array(Image.open(sample_img))
                sample_gt = np.array(Image.open(sample_gt))
            (h, w) = np.shape(sample_img)
            (hr, wr) = np.shape(sample_gt)
            scale = hr // h

            for key in weight_dict.keys():
                if subdir in key and tissue in key:
                    weight_value = weight_dict[key] * (1 / len(os.listdir(os.path.join(tissue_path, 'training_input'))))
                    if '10_' not in subdir:
                        weight_value = weight_value * 2
            for file in os.listdir(os.path.join(tissue_path, 'training_input')):
                img_list.append(os.path.join(tissue_path, 'training_input', file))
                weighted_values.append(weight_value)
                scale_list.append(scale)

                assert os.path.exists(os.path.join(tissue_path, 'training_gt', file)), print(
                    os.path.join(tissue_path, 'training_gt', file))
                gt_list.append(os.path.join(tissue_path, 'training_gt', file))

                if '10_' not in subdir:
                    img_list.append(os.path.join(tissue_path, 'training_input', file))
                    weighted_values.append(weight_value)
                    gt_list.append(os.path.join(tissue_path, 'training_gt', file))
                    scale_list.append(scale)
    return img_list, gt_list, weighted_values, scale_list


def get_selected_rotNet_train_list(root_dir, finetune_data, weight_dict, mid=False, train=True):
    if train:
        prefix = 'training'
    else:
        prefix = 'validate'
    img_list = []
    gt_list = []
    weighted_values = []
    scale_list = []
    for subdir in os.listdir(root_dir):
        if finetune_data in subdir:
            continue
        if mid:
            if '10_' not in subdir:
                continue
        else:
            if '10_' in subdir:
                continue
        subdir_path = os.path.join(root_dir, subdir)
        for tissue in os.listdir(subdir_path):
            tissue_path = os.path.join(subdir_path, tissue)
            if len(os.listdir(os.path.join(tissue_path, prefix + '_input'))) == 0:
                continue
            # print(os.path.join(tissue_path, prefix+'_input'))
            sample_img = os.path.join(tissue_path, prefix + '_input',
                                      os.listdir(os.path.join(tissue_path, prefix + '_input'))[0])
            sample_gt = os.path.join(tissue_path, prefix + '_gt', os.path.basename(sample_img))
            if sample_img.endswith('.tif'):
                sample_img = imread(sample_img).astype(np.float)
                sample_gt = imread(sample_gt).astype(np.float)

            elif sample_img.endswith('.png'):
                sample_img = np.array(Image.open(sample_img))
                sample_gt = np.array(Image.open(sample_gt))
            (h, w) = np.shape(sample_img)
            (hr, wr) = np.shape(sample_gt)
            scale = hr // h

            for key in weight_dict.keys():
                # print(key)
                if not mid and subdir in key and tissue in key:
                    weight_value = weight_dict[key] * (
                            1 / len(os.listdir(os.path.join(tissue_path, prefix + '_input'))))
                if mid and tissue in key:
                    weight_value = weight_dict[key] * (
                            1 / len(os.listdir(os.path.join(tissue_path, prefix + '_input'))))
            for file in os.listdir(os.path.join(tissue_path, prefix + '_input')):
                img_list.append(os.path.join(tissue_path, prefix + '_input', file))
                weighted_values.append(weight_value)
                scale_list.append(scale)

                assert os.path.exists(os.path.join(tissue_path, prefix + '_gt', file)), print(
                    os.path.join(tissue_path, prefix + '_gt', file))
                gt_list.append(os.path.join(tissue_path, prefix + '_gt', file))

                if '10_' not in subdir:
                    img_list.append(os.path.join(tissue_path, prefix + '_input', file))
                    weighted_values.append(weight_value)
                    gt_list.append(os.path.join(tissue_path, prefix + '_gt', file))
                    scale_list.append(scale)
    return img_list, gt_list, weighted_values, scale_list


def get_valid_list(root_dir, finetune_data):
    img_list = []
    gt_list = []
    for subdir in os.listdir(root_dir):
        if finetune_data in subdir:
            continue
        subdir_path = os.path.join(root_dir, subdir)
        for tissue in os.listdir(subdir_path):
            tissue_path = os.path.join(subdir_path, tissue)
            for file in os.listdir(os.path.join(tissue_path, 'validate_input')):
                img_list.append(os.path.join(tissue_path, 'validate_input', file))
                if not os.path.exists(os.path.join(tissue_path, 'validate_gt', file)):
                    print(os.path.join(tissue_path, 'validate_gt', file))
                assert os.path.exists(os.path.join(tissue_path, 'validate_gt', file)), print(
                    os.path.join(tissue_path, 'validate_gt', file))
                gt_list.append(os.path.join(tissue_path, 'validate_gt', file))
    return img_list, gt_list


def adjust_learning_rate(epoch, args, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs_list = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs_list.append(int(it))
    # steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs_list))
    # if steps > 0:
    #     new_lr = opt.lr * (opt.lr_decay_rate ** steps)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = new_lr
    lr = args.lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Mixup:
    def __init__(self, bs=16, beta=0.15, choice_thresh=0.3):
        self.beta = beta
        self.choice_thresh = choice_thresh

    @torch.no_grad()
    def __call__(self, lr, hr):
        bs = lr.shape[0]
        beta = Beta(torch.zeros(bs) + self.beta, torch.zeros(bs) + self.beta)
        betas = beta.sample().to(lr.device)
        perm = torch.randperm(lr.shape[0])
        lr_perm, hr_perm = lr[perm], hr[perm]
        choices = torch.rand(lr.shape[0])
        idx = torch.where(choices > self.choice_thresh)
        betas[idx] = 1.  # only choice_thresh% of samples in batch will be mixed
        lr = lr * betas.view(-1, 1, 1, 1) + lr_perm * (1 - betas.view(-1, 1, 1, 1))
        hr = hr * betas.view(-1, 1, 1, 1) + hr_perm * (1 - betas.view(-1, 1, 1, 1))
        return lr, hr


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                         (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def fluore_to_tensor(pic):
    """Convert a ``PIL Image`` to tensor. Range stays the same.
    Only output one channel, if RGB, convert to grayscale as well.
    Currently data is 8 bit depth.

    Args:
        pic (PIL Image): Image to be converted to Tensor.
    Returns:
        Tensor: only one channel, Tensor type consistent with bit-depth.
    """
    if not (_is_pil_image(pic)):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        # all 8-bit: L, P, RGB, YCbCr, RGBA, CMYK
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)

    img = img.view(pic.size[1], pic.size[0], nchannel)

    if nchannel == 1:
        img = img.squeeze(-1).unsqueeze(0)
    elif pic.mode in ('RGB', 'RGBA'):
        # RBG to grayscale:
        # https://en.wikipedia.org/wiki/Luma_%28video%29
        ori_dtype = img.dtype
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
        img = (img[:, :, [0, 1, 2]].float() * rgb_weights).sum(-1).unsqueeze(0)
        img = img.to(ori_dtype)
    else:
        # other type not supported yet: YCbCr, CMYK
        raise TypeError('Unsupported image type {}'.format(pic.mode))

    return img


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def tensor2im(image_tensor, imtype=np.uint8, normalize=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().detach().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


def mixup_data(x, alpha=1.0, index=None, lam=None, ):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if lam is None:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = lam

    lam = max(lam, 1 - lam)
    batch_size = x.size()[0]
    if index is None:
        index = torch.randperm(batch_size).cuda()
    else:
        index = index

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, lam, index


def MMD_Max(x, y, kernel):
    mulitiple = x.shape[0] // 64
    if mulitiple == 0:
        return MMD(x, y, kernel)
    mmd = 0.0
    for i in range(mulitiple):
        mmd += MMD(x[i * 128: (i + 1) * 128], y[i * 128: (i + 1) * 128], kernel)
    mmd /= mulitiple
    return mmd


def MMD(x, y, kernel):
    if x.shape[0] == 0:
        return 0
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    #  print('inner', x.shape, y.shape)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda())
    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
    return torch.mean(XX + YY - 2. * XY)


def generate_crop(img_list, patch_size=512, stride=256):
    sample_img = img_list[0]
    if sample_img.endswith('.tif'):
        img = Image.fromarray(imread(sample_img).astype(np.float))
    elif sample_img.endswith('.png'):
        img = Image.open(sample_img)
    width, height = img.size
    print(f'sample image size{width}, {height}')
    if width > patch_size:
        print('generating the crop')
        cropped_list = []
        num_patches = 1
        for img in img_list:
            if img.endswith('.tif'):
                img = imread(img).astype(np.float)

            elif img.endswith('.png'):
                img = np.array(Image.open(img))
                (h, w) = np.shape(img)
            h, w = img.shape
            idx_x = 0
            while (idx_x + patch_size < h):
                idx_y = 0
                while (idx_y + patch_size < w):
                    patch = img[idx_x:idx_x + patch_size, idx_y:idx_y + patch_size]
                    cropped_list.append(patch)
                    num_patches = num_patches + 1
                    idx_y = idx_y + stride
                idx_x = idx_x + stride
        return cropped_list
    else:
        return img_list
