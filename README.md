# (ICCV 2025) GT-Mean Loss: A Simple Yet Effective Solution for Brightness Mismatch in Low-Light Image Enhancement.

We use Retinexformer as an example to quickly verify the effect of GT-mean loss.  

See .\basicsr\models\losses\losses.py for the implementation of GT-mean loss

## 1. Create Environment

- Make Conda Environment
```
conda create -n Retinexformer python=3.7
conda activate Retinexformer
```

- Install Dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```


&nbsp;

## 2. Prepare Dataset

<details close>
    <summary><b> The LOL-v1 and LOL-v2 datasets are organized as follows:  </b></summary>

```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...

```
</details>


## 3. Testing

```shell
# LOL-v1
# Retinexformer
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1
# Retinexformer with GT-mean loss
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1_GTmean.pth --dataset LOL_v1

# LOL-v2-real
# Retinexformer
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real.pth --dataset LOL_v2_real
# Retinexformer with GT-mean loss
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_real.yml --weights pretrained_weights/LOL_v2_real_GTmean.pth --dataset LOL_v2_real

# LOL-v2-synthetic
# Retinexformer
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic.pth --dataset LOL_v2_synthetic
# Retinexformer with GT-mean loss
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml --weights pretrained_weights/LOL_v2_synthetic_GTmean.pth --dataset LOL_v2_synthetic
```


## 4. Training
```shell
# LOL-v1,Retinexformer
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v1.yml
# LOL-v1, Retinexformer with GT-mean loss
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v1_GTmeanLoss.yml

# LOL-v2-real, Retinexformer
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_real.yml
# LOL-v2-real, Retinexformer with GT-mean loss
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_real_GTmeanLoss.yml

# LOL-v2-syntheticl, Retinexformer
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_synthetic.yml
# LOL-v2-syntheticl, Retinexformer with GT-mean loss
python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v2_synthetic_GTmeanLoss.yml
```

## Citation
```
@InProceedings{gtmeanloss_liao,
    author = {Jingxi, Liao and Shijie, Hao and Richang, Hong and Meng, Wang},
    title = {GT-Mean Loss: A Simple Yet Effective Solution for Brightness Mismatch in Low-Light Image Enhancement},
    booktitle = {ICCV},
    year = {2025}
}
```
