import numpy as np
import h5py
import glob
import os
import cv2


class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)

    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype

    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)

    """

    def __init__(self, datapath, dataset, shape, dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.i = 0

        with h5py.File(self.datapath, mode='w') as h5f:
            self.dset = h5f.create_dataset(
                dataset,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len,) + shape)

    def append(self, values):
        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1,) + self.shape)
            dset[self.i] = [values]
            self.i += 1
            h5f.flush()


def generate_h5f(data_path, folder_name, patch_size, stride, number_of_training_images=240):
    '''
    This function generates all h5 files and stores them in 'net_data/' with name: f'{folder_name}_{patch_size}_{stride}'

        data_path: where the image folders are stored, each folder is avg1, avgX etc, +sim
        folder_name: picks between avg1, sim, etc
        number_of_training_images: how many images to use for the training set

        Ex: data_path = '../data/all' + folder_name = 'avg1'
        Ex: data_path = '../results/BM3D' + folder_name = 'avg1'
    '''

    print(f'Creating training h5f, for {folder_name}...')

    # OLD normalization: files = glob.glob(os.path.join(data_path, folder_name, '*.png'))
    files = glob.glob(os.path.join(data_path, folder_name, '*.npy'))
    files.sort()
    if number_of_training_images > len(files):
        raise NotImplementedError(f'Maximum available images: {len(files)}.')

    train_file_name = f'{folder_name}_{patch_size}_{stride}'
    patch_shape = (1, patch_size, patch_size)
    hdf5 = HDF5Store(datapath=os.path.join('./net_data/', train_file_name) + '.h5', dataset='data', shape=patch_shape)

    num_patches = 0
    for idx in range(number_of_training_images):
        # OLD normalization: img = cv2.imread(files[idx], cv2.IMREAD_GRAYSCALE)
        img = np.load(files[idx])
        (h, w) = np.shape(img)

        idx_x = 0
        while (idx_x + patch_size < h):
            idx_y = 0
            while (idx_y + patch_size < w):
                patch = img[idx_x:idx_x + patch_size, idx_y:idx_y + patch_size]
                hdf5.append(patch / 255.)
                num_patches = num_patches + 1

                idx_y = idx_y + stride

            idx_x = idx_x + stride

    print('%d patches generated, and saved.' % (num_patches))


data_path = '/mnt/sdc/cxlu/16/raw'
number_of_training_images = 240

for folder_name in ['avg400', 'sim', 'avg1', 'avg2', 'avg4', 'avg8', 'avg16']:

    patch_size = 64
    stride = 32
    if folder_name == 'sim':
        patch_size = patch_size * 2
        stride = stride * 2

    generate_h5f(data_path, folder_name, patch_size, stride, number_of_training_images)
