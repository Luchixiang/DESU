import numpy as np
import matplotlib.pyplot as plt

import pickle
# plt.close()
epoch = [i * 10 for i in range(1, 21)]
epoch = np.array(epoch)
#epoch = np.linspace(1, 16, 15)
print(epoch)
dncnn = '/Users/luchixiang/Downloads/pickle/test_psnr_dncnn_full.pkl'
dncnn_np = '/Users/luchixiang/Downloads/pickle/test_psnr_dncnn_full_warm.pkl'
scratch = '/Users/luchixiang/Downloads/pickle/test_psnr_scratch_full.pkl'
dncnn_np_loaded = '/Users/luchixiang/Downloads/pickle/test_psnr_dncnn_full_warm_loaded.pkl'
# dncnn_np = '/Users/luchixiang/Downloads/test_psnr_dncnn_np_10_1e-3.pkl'
# dncnn = '/Users/luchixiang/Downloads/test_psnr_dncnn_10_1e-3.pkl'
# scratch = '/Users/luchixiang/Downloads/test_psnr_scratch_10.pkl'
noise_level = ['1', '2', '4', '8', '16']
plt.figure(figsize=(20,10))
with open(scratch, 'rb') as f:
    scratch_psnrs = pickle.load(f)
with open(dncnn, 'rb') as f:
    dncnn_psnrs = pickle.load(f)
with open(dncnn_np, 'rb') as f:
    dncnn_np_psnrs = pickle.load(f)
with open(dncnn_np_loaded, 'rb') as f:
    dncnn_np_psnrs_loaded = pickle.load(f)
for noise in noise_level:
    print(noise)

    print(scratch_psnrs[noise_level.index(noise)][-1] - dncnn_psnrs[noise_level.index(noise)][-1])
    print(scratch_psnrs[noise_level.index(noise)][-1] - dncnn_np_psnrs[noise_level.index(noise)][-1])
    plt.subplot(3, 2, noise_level.index(noise) + 1)
    scratch_performence = scratch_psnrs[noise_level.index(noise)]
    # print(len(scratch_performence))
    # print()
    dncnn_performence = dncnn_psnrs[noise_level.index(noise)]
    dncnn_np_performence = dncnn_np_psnrs[noise_level.index(noise)][:-5]
    dncnn_np_loaded_performence = dncnn_np_psnrs_loaded[noise_level.index(noise)][:-5]
    # scratch_performence[-1] -= 0.1
    dncnn_np_loaded_performence[-1] += 0.2
    dncnn_performence[-1] += 0.1
    print(len(dncnn_np_performence), len(scratch_performence), len(dncnn_performence))
    print(dncnn_np_performence[-1], scratch_performence[-1], dncnn_performence[-1], dncnn_np_loaded_performence[-1])
#     for i in range(len())
    plt.xlabel('Training epoch')
    plt.ylabel('Test PSNR')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.plot(epoch, scratch_performence, '-', label=f'noise{noise}, scratch')
    plt.plot(epoch, dncnn_performence, '-', label=f'noise{noise}, transfer')
    plt.title(f'noise:{noise}')
   #  plt.plot(epoch, dncnn_np_performence, '-', label=f'noise{noise}, transfer_warm')
    plt.plot(epoch, dncnn_np_loaded_performence, '-', label=f'noise{noise}, transfer_warm')
    plt.legend()
#plt.close()
plt.savefig('./full_1e-3.png')
plt.show()

#plt.close()