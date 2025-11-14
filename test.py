import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

from models.EccoMamba.EccoMamba import EccoMamba
from models.vmunet.vmunet import VMUNet
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main():
    # get configs from setting_config and command line arguments
    config = setting_config
    config.add_argument_config()
    config.set_datasets()
    config.set_opt_sch()


    print('#----------Creating logger----------#')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')

    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    test_model_path = config.best_model_path
    if not os.path.exists(test_model_path):
        raise Exception('test model is not exist!')
    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    model = EccoMamba(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    # model = VMUNet(
    #         num_classes=model_cfg['num_classes'],
    #         input_channels=model_cfg['input_channels'],
    #         depths=model_cfg['depths'],
    #         depths_decoder=model_cfg['depths_decoder'],
    #         drop_path_rate=model_cfg['drop_path_rate'],
    #         load_ckpt_path=model_cfg['load_ckpt_path'],
    #     )
    model.load_from()
    model = model.cuda()

    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
    else:
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print('#----------Preparing dataset----------#')
    if config.datasets_name == "isic2017" or config.datasets_name == "isic2018":
        test_dataset = config.datasets(path_Data = config.data_path, train = False, Test = True)
        test_loader = DataLoader(test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True, 
                                    num_workers=config.num_workers,
                                    drop_last=True)
    elif config.datasets_name == "synapse" or config.datasets_name == "acdc":
        val_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if config.distributed else None
        val_loader = DataLoader(val_dataset,
                                batch_size=1, # if config.distributed else config.batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers, 
                                sampler=val_sampler,
                                drop_last=True)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()
    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    max_dice = 0
    max_dsc  = 0.88


    print('#----------Testing----------#')
    checkpoint = torch.load(test_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()  # Set the model to evaluation mode
    if config.datasets_name == "isic2017" or config.datasets_name == "isic2018":
        with torch.no_grad():  # Disable gradient calculation during testing
            loss, miou, f1_or_dsc = test_one_epoch(
                test_loader,
                model,
                criterion,
                logger,
                config,
            )
    elif config.datasets_name == "synapse" or config.datasets_name == "acdc":
        with torch.no_grad(): 
            mean_dice, mean_hd95 = test_sy_ac(
                    val_dataset,
                    val_loader,
                    model,
                    logger,
                    config,
                    test_save_path=outputs,
                    val_or_test=True
                )


if __name__ == '__main__':
    main()