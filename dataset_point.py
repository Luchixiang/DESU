import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import random
from torch.utils.data import Dataset
from util import fluore_to_tensor


class BioDataset(Dataset):

    def initialize(self, train=True):
        # if train:
        #     f = open('/home/cxlu/c2f/pix2pixHD/A2B_point_train.json')
        # else:
        #     f = open('/home/cxlu/c2f/pix2pixHD/A2B_point_test.json')
        if train:
            f = open('./A2B_point_train.json')
        else:
            f = open('./A2B_point_test.json')
        contents = []
        with open('not_background.txt') as fp:
            for line in fp.readlines():
                contents.append(line.strip('\n'))
        transform_List = [transforms.Resize((256, 256)), fluore_to_tensor,
                          transforms.Lambda(lambda x: x.float().div(255).sub(0.5))]  # todo 增加flip,maybe rotate
        self.A_transform = transforms.Compose(transform_List)
        transform_List = [transforms.Resize((512, 512)), fluore_to_tensor,
                          transforms.Lambda(lambda x: x.float().div(255).sub(0.5))]  # todo 增加flip,maybe rotate
        self.B_transform = transforms.Compose(transform_List)

        self.a2b_dict = json.load(f)
        self.A_paths = []
        import random
        for key in self.a2b_dict.keys():
            ket_replace = self.a2b_dict[key].replace('/mnt/sdc/cxlu/c2f/point', '/Users/Luchixiang/Downloads')
            # print(ket_replace)
            if ket_replace in contents:

                self.A_paths.append(key)
            # if len(self.A_paths) == 1000:
            #     break
        f.close()

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        ### input A (label maps)
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        ### input B (real images)

        # if not self.opt.isTrain:
        #     # print(A_path)
        #     B_path = self.a2b_dict[A_path.replace('coarse_imgs_test', 'coarse_imgs')]
        # else:
        #     B_path = self.a2b_dict[A_path]
        B_path = self.a2b_dict[A_path]
        B = Image.open(B_path).convert('RGB')
        A_tensor, B_tensor = self.aug(A, B)
        # transform_B = get_transform(self.opt, params)
        # B_tensor = transform_B(B)
        # print('in dataset:', A_tensor.shape, B_tensor.shape)
        return A_tensor, B_tensor, A_path
        # input_dict = {'label': A_tensor, 'image': B_tensor,
        #               'path': A_path}

    def aug(self, A, B):
        def __flip(img):
            return img.transpose(Image.FLIP_LEFT_RIGHT)

        def __rotate(img):
            return img.transpose(Image.ROTATE_180)

        # if random.random() < 0.5 and self.opt.isTrain:
        #     A = __flip(A)
        #     B = __flip(B)
        # if random.random() < 0.5 and self.opt.isTrain:
        #     A = __rotate(A)
        #     B = __rotate(B)
        A = self.A_transform(A)
        B = self.B_transform(B)
        return A, B

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
