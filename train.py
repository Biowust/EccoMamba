import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *
import matplotlib.pyplot as plt
from models.EccoMamba.EccoMamba import EccoMamba
from models.vmunet.vmunet import VMUNet

from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"
from datasets.dataset import RandomGenerator
from utils import *
from configs.config_setting import setting_config
from dataset_Monu import RandomGenerator_Monu,ValGenerator,ImageToImage2D,train_one_epoch_MoNu,val_one_epoch_MoNu
import warnings
warnings.filterwarnings("ignore")

def worker_init_fn(worker_id):
    random.seed(42 + worker_id)
def main():
    # get configs from setting_config and command line arguments
    config = setting_config
    config.add_argument_config()
    config.set_datasets()
    config.set_opt_sch()

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    if config.datasets_name == "isic2017" or config.datasets_name == "isic2018":
        train_dataset = config.datasets(path_Data = config.data_path, train = True)
        train_loader = DataLoader(train_dataset,
                                    batch_size=config.batch_size, 
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=config.num_workers)
        val_dataset = config.datasets(path_Data = config.data_path, train = False)
        val_loader = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True, 
                                    num_workers=config.num_workers,
                                    drop_last=True)
        test_dataset = config.datasets(path_Data = config.data_path, train = False, Test = True)
        test_loader = DataLoader(test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True, 
                                    num_workers=config.num_workers,
                                    drop_last=True)
    elif config.datasets_name == "synapse" or config.datasets_name == "acdc":
        train_dataset = config.datasets(base_dir=config.data_path, list_dir=config.list_dir, split="train",
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[config.input_size_h, config.input_size_w])]))
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if config.distributed else None
        train_loader = DataLoader(train_dataset,
                                    batch_size=config.batch_size//gpus_num if config.distributed else config.batch_size, 
                                    shuffle=(train_sampler is None),
                                    pin_memory=True,
                                    num_workers=config.num_workers,
                                    sampler=train_sampler)

        val_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if config.distributed else None
        val_loader = DataLoader(val_dataset,
                                batch_size=1, # if config.distributed else config.batch_size,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers, 
                                sampler=val_sampler,
                                drop_last=True)
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
    #     num_classes=model_cfg['num_classes'],
    #     input_channels=model_cfg['input_channels'],
    #     depths=model_cfg['depths'],
    #     depths_decoder=model_cfg['depths_decoder'],
    #     drop_path_rate=model_cfg['drop_path_rate'],
    #     load_ckpt_path=model_cfg['load_ckpt_path'],
    # )
    model.load_from()
    model = model.cuda()

    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
    else:
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    scaler = GradScaler()
    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    max_dice = 0.8
    max_dsc  = 0.88
    max_iou = 0.0
    best_epoch= -1
    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    print('#----------Training----------#')

    loss_values = []
    for epoch in tqdm(range(start_epoch, config.epochs + 1)):

        torch.cuda.empty_cache()
        train_sampler.set_epoch(epoch) if config.distributed else None
        if config.datasets_name == "isic2017" or config.datasets_name == "isic2018":
            train_loss = train_one_epoch_isic(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                logger,
                config,
                scaler=scaler
            )
        elif config.datasets_name == "synapse" or config.datasets_name == "acdc":
            train_loss = train_one_epoch_sy_ac(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )
        elif config.datasets_name == "monuseg":
            optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer2, T_0=10, T_mult=1, eta_min=1e-4)
            train_loss,_,_ = train_one_epoch_MoNu(train_loader, model, criterion, optimizer, epoch, None, logger,config)
            train_loss = train_loss.detach().cpu().numpy()
        loss_values.append(train_loss)
            
        plt.figure()
        ax = plt.gca()
        ax.plot(loss_values)
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.savefig(os.path.join(config.work_dir, f"loss.png"))
        plt.close()


        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': train_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

        if config.datasets_name == "isic2017" or config.datasets_name == "isic2018":
            if epoch > 30 and epoch % 20 == 0:
                loss, miou, f1_or_dsc = test_one_epoch(
                        test_loader,
                        model,
                        criterion,
                        logger,
                        config,
                    )
                if f1_or_dsc > max_dsc:
                    print('---saving best model---')
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir,f'epoch{epoch}-miou{miou:.4f}-dsc{f1_or_dsc:.4f}.pth'))
                    max_dsc = f1_or_dsc
        elif config.datasets_name == "synapse" or config.datasets_name == "acdc":
            if epoch > 100 and epoch % 20 == 0:
                mean_dice, mean_hd95 = val_one_epoch_(
                val_dataset,
                val_loader,
                model,
                epoch,
                logger,
                config,
                test_save_path=outputs,
                val_or_test=True
            )
                if mean_dice > max_dice:
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir,f'epoch:{epoch}-mean_dice:{mean_dice:.4f}-mean_hd95:{mean_hd95:.4f}.pth'))
                    max_dice = mean_dice
        elif config.datasets_name == "monuseg":
            if epoch > 50 and epoch % 20 == 0:
                val_loss, val_dice, val_iou = val_one_epoch_MoNu(val_loader, model, criterion, logger, config)
                
                if val_dice > max_dice:
                    if epoch+1 > 1:
                        logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
                        logger.info('\t mean iou: {:.4f}'.format(val_iou))
                        max_dice = val_dice
                        max_iou = val_iou
                        best_epoch = epoch + 1
                        if max_dice > 0.8:
                            save_checkpoint({'epoch': epoch,
                                            'best_model': True,
                                            'model_state_dict': model.state_dict(),
                                            'max_dice':max_dice,
                                            'val_loss': val_loss,
                                            'max_iou':max_iou,
                                            'optimizer_state_dict': optimizer.state_dict()}, config.work_dir)

                else:
                    logger.info('\t Mean dice:{:.4f} does not increase, '
                                'the best is still: {:.4f} and {:.4f} in epoch {}'.format(val_dice,max_dice, max_iou, best_epoch))
def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    dice = state['max_dice']  # floatq
    

    filename = f'epoch_{epoch}_{dice}.pth'
    save_mode_path = os.path.join(save_path, filename)
    torch.save(state, save_mode_path)

if __name__ == '__main__':

    main()

