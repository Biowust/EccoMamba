import torch.optim
from dataset_Monu import ValGenerator,ImageToImage2D,train_one_epoch_MoNu
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from utils import *
import cv2
from pathlib import Path
import argparse
from configs.config_setting import setting_config
from GradCAM import show_cam_on_image
from PIL import Image
from skimage import morphology
from GradCAM import GradCAM
from models.EccoMamba.EccoMamba import EccoMamba
from models.vmunet.vmunet import VMUNet
import numpy as np
import torch.nn as nn


def show_image_with_dice(predict_save, labs, save_path, img_RGB=None):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    
    # 添加调试信息
    print(f"In show_image_with_dice:")
    print(f"predict_save shape: {predict_save.shape}, min: {predict_save.min()}, max: {predict_save.max()}")
    print(f"labs shape: {labs.shape}, min: {labs.min()}, max: {labs.max()}")
    
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
  
    predict_save = cv2.pyrUp(predict_save,(448,448))
    predict_save = cv2.resize(predict_save,(2000,2000))
    cv2.imwrite(save_path,predict_save * 255)
    
    if img_RGB is not None:
        if len(tmp_lbl.shape) > 2:
            tmp_lbl = tmp_lbl.squeeze()  # 如果是3D，压缩到2D
        if len(tmp_3dunet.shape) > 2:
            tmp_3dunet = tmp_3dunet.squeeze()  # 如果是3D，压缩到2D

        output_size = (1000, 1000)  # 可以根据需要调整
        
        gt_binary = (cv2.resize(tmp_lbl, output_size) > 0.5).astype(np.uint8) * 255
        pred_binary = (cv2.resize(tmp_3dunet, output_size) > 0.5).astype(np.uint8)
        
        output_img = cv2.cvtColor(gt_binary, cv2.COLOR_GRAY2RGB)
        

        kernel_alt = np.ones((2, 2), np.uint8)
        pred_edges = cv2.morphologyEx(pred_binary, cv2.MORPH_GRADIENT, kernel_alt) * 255

        kernel = np.ones((3, 3), np.uint8)  # 调整kernel大小可以控制边缘粗细
        pred_edges = cv2.dilate(pred_edges, kernel, iterations=1)  # 适当的迭代次数
        
        output_img[pred_edges > 0, 0] = 0    # B通道
        output_img[pred_edges > 0, 1] = 0    # G通道
        output_img[pred_edges > 0, 2] = 255  # R通道
        
        edge_on_label_path = save_path.rsplit('.', 1)[0] + '_edge_on_label.jpg'
        cv2.imwrite(edge_on_label_path, output_img)
        
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        plt.title('GT Label with Prediction Edges (Red)')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Ground Truth'),
            Patch(facecolor='red', label='Prediction Edge')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        plt.axis('off')
        
        # 保存带有图例的图像
        legend_path = save_path.rsplit('.', 1)[0] + '_edge_on_label_with_legend.jpg'
        plt.savefig(legend_path)
        plt.close()

    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_img = input_img.to(device)
    
    model.eval()

    output = model(input_img) 
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (args.img_size, args.img_size))
    save_path = vis_save_path.rsplit('.', 1)[0] + '_predict.jpg'
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=save_path, img_RGB=img_RGB)

    return dice_pred_tmp, iou_tmp



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    test_num = 14

    model_path = '/data/ljc/source/outputs/VMUnet/monuseg/epoch_300_0.9078500270843506.pth'
    vis_path = '/data/ljc/source/outputs/VMUnet/monuseg/test/vis/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str,
                    default='/data/ljc/source/data/MoNuSeg/Val_Folder/', help='root dir for validation volume data')
    parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')     
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--cfg', type=str, default='/data/ljc/source/SWMA-UNet/configs/swin_tiny_patch4_window7_224_lite.yaml',metavar="FILE", help='path to config file', )    
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )      
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--batch_size', type=int, default=36,help='batch_size per gpu') 
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--model_name', type=str,default='SwmaUnet')                
    args = parser.parse_args()

    # config = get_config(args)
    config = setting_config
    checkpoint = torch.load(model_path, map_location='cuda')
    # model = swma_unet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    model_cfg = config.model_config
    # model = EccoMamba(
    #     num_classes=model_cfg['num_classes'],
    #     input_channels=model_cfg['input_channels'],
    #     depths=model_cfg['depths'],
    #     depths_decoder=model_cfg['depths_decoder'],
    #     drop_path_rate=model_cfg['drop_path_rate'],
    #     load_ckpt_path=model_cfg['load_ckpt_path'],
    # )
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
    )
    model = torch.nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    # Handle DataParallel prefix issue
    state_dict = checkpoint['model_state_dict']
    if isinstance(state_dict, dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]  # Remove 'module.' prefix
            else:
                new_key = k
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[args.img_size, args.img_size])
    test_dataset = ImageToImage2D(args.test_path, tf_test,image_size=args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    list=[]
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            
            # 确保lab是2D数组
            if len(lab.shape) > 2:
                lab = np.squeeze(lab)  # 压缩多余的维度
            
            img_lab = np.reshape(lab, (lab.shape[0], lab.shape[1])) * 255
            
            # 获取原始RGB图像用于叠加显示
            img_RGB = arr[0].transpose(1, 2, 0)  # 将通道维度移到最后
            if img_RGB.shape[2] == 1:  # 如果是单通道图像，转换为3通道
                img_RGB = np.repeat(img_RGB, 3, axis=2)
            img_RGB = (img_RGB * 255).astype(np.uint8)  # 转换为0-255的RGB

            original_name = names[0]  
            base_name = Path(original_name).stem  

            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = args.img_size, args.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)

            save_path = Path(vis_path) / f"{base_name}_lab.jpg"  
            plt.savefig(save_path, dpi=300)
            plt.close()

            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(
                model, input_img, img_RGB, lab,
                str(Path(vis_path) / base_name),  
                dice_pred=dice_pred, dice_ens=dice_ens
            )
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            list.append(dice_pred_t)
            torch.cuda.empty_cache()
            pbar.update()

    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)
    print(list)
