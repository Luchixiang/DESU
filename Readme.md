## Readme

### Pretraining

To run the pretraining stage, you should firstly download the DESU dataset. The dataset can be downlaoded from [pretraining](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3590540_connect_hku_hk/Eq_di1Iuv5ZEgVUq0JdhimUBZqTOwVbKsUWSb5Y70PzxGQ?e=KKYXXe) and unzip all the compressed files.

```
git clone https://github.com/Luchixiang/DESU
or download from moodle website
cd desu
```

**ensure that you have 2 gpu**

1. pretrain on all the datasets

```bash
python main_whole.py --data /mnt/sdb/cxlu/SR_Data_processed --b 256  --epoch 200 --lr 1e-3 --output ./weight_pretrain --mixup --world_size 1 --rank 0 --gpus 0,1 --local_rank 1 --num_worker 12 --finetune 10_processed/WideField_BPAE_R --mixup
```

please replace the '--data' with your root of pretraining dataset

--finetune is the downstream task and this dataset is exclued from the pretraining dataset.



3. use MMD-DS to select data and pretrain

```
step1: python select_data_mmd.py -- data '/mnt/sdb/cxlu/SR_Data_processed'
```

please replace  the '--data' with your root of pretraining dataset

````
step2: python main_mmd.py --data /mnt/sdb/cxlu/SR_Data_processed --b 256  --epoch 200 --lr 1e-3 --output ./weight_pretrain --world_size 1 --rank 0 --gpus 0,1 --local_rank 1 --num_worker 8 --finetune 10_processed/WideField_BPAE_R --pickle_file ./mmd.pkl
````



### finetune

#### finetune w2s dataset

1. 

```
cd finetune/w2s/denoise
```

2. fine-tune the model using the pretrained weight

```
python train.py --model fullwarm --warm --img_avg 2 --weight ./weight_pretrain/best.pt --data_path /mnt/sdb/cxlu/SR_Data_processed/16_denoise/tissue/training_input
```

please repalce the --weight with your filepath saving the pretraining weight and --data_path with the path of w2s dataset(where we named 16_denoise in our pretraining dataset ). --img_avg is the noise level you want to use. Please choose ont from [1, 2, 4, 8, 16] and 1 is the most noisy image.

The weight of fine-tuned netowork is save is '../net_data/trained_denoisers_avg{opt.img_avg}_ {opt.model}/{opt.net}_{opt.img_avg}/'. For example, for the last command, the weight is saved in '--/net_data/trained_denoisers_avg2_fullwarm/D_2/'

3. test the model

```
python test.py --model_dir '--/net_data/trained_denoisers_avg2_fullwarm/D_2/' --epcoh 49
```

Replace the --model_dir with the path you saved the fine-tuned weight as indicated in last step.





#### finetune FMD dataset

1.

```
cd finetune/denoising-fluorescence/denoising
```



2. Download FMD dataset from: https://drive.google.com/drive/folders/1aygMzSDdoq63IqSk-ly8cMq0_owup8UM and unzip them

3. fine-tune the model using the pretrained weight

```
python train_dncnn.py --exp-dir ./experiment --data_root /mnt/sdb/cxlu/SR_Data/10/fmdd --weight ./weight_pretrain/best.pt
```

Please replace --data_root with your fmd dataset dpath and --weight with your pretrained weight path

the model will be saved in {opt.exp_dir} + '/' + 'dncnn' + '/' + {date} + 'dncnn_noise_train[1, 2, 4, 8, 16] _ test[1]_captures50_four_crop_epochs400_bs8_lr0.0001' where {date} is the date you trained model. For example: Jul_22.



4. test the model

```
python benchmark.py -- pretrained-dir ./experiment_dissimilar/dncnn_np/Jul_22/dncnn_noise_train[1, 2, 4, 8, 16]_test[1]_captures50_four_crop_epochs400_bs8_lr0.0001 --data_root /mnt/sdb/cxlu/SR_Data/10/fmdd
```

replace --pretrained-dir with the path indicated in the last step and --data_root with your fmd dataset place.