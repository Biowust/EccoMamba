# EccoMamba

This repository is the official implementation of EccoNet. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## DataSet
You can download four processed datasets: ISIC 2018, ISIC 2017, ACDC, and Synapse. [click this](https://zenodo.org/records/14626096)

## Training

To train the model(s) in the paper, run this command:

```train
cd EccoNet
python train.py --datasets_name <dataset name> --epochs 1000 --batch_size 36 --work_dir <output dir>
```


## Evaluation

You can download the best weights files for the four datasets for verification here.[click this]([https://zenodo.org/records/14626096](https://zenodo.org/records/16451011))

To evaluate my model, run:
```eval
python test.py --datasets_name <dataset name>  --batch_size 36 --work_dir <output dir> --best_model_path <the best weighted path>
```


## Pre-trained Models

You can download pretrained models here:

- [VMamba](https://github.com/MzeroMiko/VMamba) 




## References
[VMamba](https://github.com/MzeroMiko/VMamba)

[VMUNet](https://github.com/JCruan519/VM-UNet?tab=readme-ov-file)

[SwinUNet](https://github.com/HuCaoFighting/Swin-Unet?tab=readme-ov-file)
