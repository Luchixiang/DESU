import numpy as np
import matplotlib.pyplot as plt

import pickle
# plt.close()
epoch = [i for i in range(1, 21)]

scratch = '/Users/luchixiang/Downloads/test_psnr.pkl'

noise_level = ['1', '2', '4', '8', '16']
plt.figure(figsize=(20,10))
with open(scratch, 'rb') as f:
    scratch_psnrs = pickle.load(f)
# with open(dncnn, 'rb') as f:
#     dncnn_psnrs = pickle.load(f)
# with open(dncnn_np, 'rb') as f:
#     dncnn_np_psnrs = pickle.load(f)
for noise in noise_level:
    plt.subplot(3, 2, noise_level.index(noise) + 1)
    scratch_performence = scratch_psnrs[noise_level.index(noise)]
    print(len(scratch_performence))
    # print()d
    # dncnn_performence = dncnn_psnrs[noise_level.index(noise)][::2]
    # dncnn_np_performence = dncnn_np_psnrs[noise_level.index(noise)][::2]
#     for i in range(len())
    plt.plot_date(epoch,scratch_performence, '-', label=f'noise{noise}, scratch')
    # plt.plot_date(epoch,dncnn_performence, '-', label=f'noise{noise}, transfer')
    # plt.plot_date(epoch,dncnn_np_performence, '-', label=f'noise{noise}, transfer_np')
    plt.legend()
#plt.close()
plt.show()
plt.savefig('test.png')
#plt.close()