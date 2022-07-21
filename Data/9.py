import numpy as np
from tifffile import imread, imwrite
import os
## generate the noisy data

def autocorrelation(x):
    """
    nD autocorrelation
    remove mean per-patch (not global GT)
    normalize stddev to 1
    value at zero shift normalized to 1...
    """
    x = (x - np.mean(x)) / np.std(x)
    x = np.fft.fftn(x)
    x = np.abs(x) ** 2
    x = np.fft.ifftn(x).real
    x = x / x.flat[0]
    x = np.fft.fftshift(x)
    return x


np_file = '/mnt/sdb/cxlu/SR_Data/9/gt.npy'
target_path = '/mnt/sdb/cxlu/SR_Data/9_processed/membrane'
if not os.path.exists(os.path.join(target_path, 'training_input')):
    os.mkdir(target_path)
    os.mkdir(os.path.join(target_path, 'training_input'))
    os.mkdir(os.path.join(target_path, 'training_gt'))
    os.mkdir(os.path.join(target_path, 'validate_input'))
    os.mkdir(os.path.join(target_path, 'validate_gt'))
X = np.load(np_file).astype(np.float32)
print(X.shape)
xautocorr = np.array([autocorrelation(_x) for _x in X])
x = xautocorr.mean(0)


def crop_square_center(x, w=20):
    a, b = x.shape
    x = x[a // 2 - w:a // 2 + w, b // 2 - w:b // 2 + w]
    return x


from scipy.ndimage import convolve

purenoise = []
noise_kernel = np.array([[1, 1, 1]]) / 3  ## horizontal correlations
a, b, c = X.shape
for i in range(a):
    noise = np.random.rand(b, c) * 1.5
    noise = convolve(noise, noise_kernel)
    purenoise.append(noise)
purenoise = np.array(purenoise)
purenoise = purenoise - purenoise.mean()

noisy_dataset = X + purenoise
inds = np.arange(X.shape[0])
np.random.shuffle(inds)
X_val = noisy_dataset[inds[:800]]
X_train = noisy_dataset[inds[800:]]
gt_val = X[inds[:800]]
gt_train = X[inds[800:]]
print(X_train.shape, X_val.shape)

for i in range(X_train.shape[0]):
    imwrite(os.path.join(target_path, 'training_input', str(i) + '.tif'), X_train[i])
    imwrite(os.path.join(target_path, 'training_gt', str(i) + '.tif'), gt_train[i])

for i in range(X_val.shape[0]):
    imwrite(os.path.join(target_path, 'validate_input', str(i) + '.tif'), X_val[i])
    imwrite(os.path.join(target_path, 'validate_gt', str(i) + '.tif'), gt_val[i])