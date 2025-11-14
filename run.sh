# train and test EccoMamba
# python /data/ljc/source/EccoMamba/train.py --datasets_name isic2018 --epochs 500 --batch_size 24 --work_dir /data/ljc/source/outputs/EccoMamba/ISIC2018_1/
python /data/ljc/source/EccoMamba/train.py --datasets_name isic2018 --epochs 500 --batch_size 24 --work_dir /data/ljc/source/outputs/EccoMamba/ISIC2018_2/
python /data/ljc/source/EccoMamba/train.py --datasets_name isic2018 --epochs 500 --batch_size 24 --work_dir /data/ljc/source/outputs/EccoMamba/ISIC2018_3/

python /data/ljc/source/EccoMamba/train.py --datasets_name isic2017 --epochs 500 --batch_size 24 --work_dir /data/ljc/source/outputs/EccoMamba/ISIC2017_1/
python /data/ljc/source/EccoMamba/train.py --datasets_name isic2018 --epochs 500 --batch_size 24 --work_dir /data/ljc/source/outputs/EccoMamba/ISIC2017_2/
python /data/ljc/source/EccoMamba/train.py --datasets_name isic2018 --epochs 500 --batch_size 24 --work_dir /data/ljc/source/outputs/EccoMamba/ISIC2017_3/

# python /data/ljc/source/EccoMamba/train.py --datasets_name isic2017 --epochs 500 --batch_size 24 --work_dir /data/ljc/source/outputs/EccoMamba/ISIC2017/

# python /data/ljc/source/EccoMamba/train.py --datasets_name isic2018 --epochs 500 --batch_size 24 --work_dir /data/ljc/source/outputs/EccoMamba/ISIC2018_1008/

# python /data/ljc/source/EccoMamba2/train.py --datasets_name isic2017 --epochs 500 --batch_size 24 --work_dir /data/ljc/source/outputs/EccoMamba/ISIC2017_decoer/

# python /data/ljc/source/EccoMamba/test.py --datasets_name isic2018 --epochs 500 --batch_size 24 --work_dir /data/ljc/source/outputs/EccoMamba/isic2018/ --best_model_path /data/ljc/source/outputs/EccoMamba/ISIC2018_best.pth

# # test vmunet
# python /data/ljc/source/EccoMamba/test.py --datasets_name isic2017 --epochs 500 --batch_size 24 --work_dir /data/ljc/source/outputs/vmunet/isic2018_to17/ --best_model_path /data/ljc/source/outputs/VMUnet/isic2018_1004/checkpoints/epoch60-miou0.8031-dsc0.8908.pth

# python /data/ljc/source/EccoMamba/train.py --datasets_name isic2018 --epochs 300 --batch_size 24 --work_dir /data/ljc/source/outputs/VMUnet/isic2018_1004/