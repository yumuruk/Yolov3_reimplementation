import sys

from data_utils import Combine_Dataset, UIEBDataset

import os
import torch
import argparse
# from model import BLUE_Net, load_model
from models.combine_model import Dual_Net, YoloV3, Darknet53
# from utils.loss import compute_loss
# import wandb
from datetime import datetime
import time
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torchmetrics.functional import peak_signal_noise_ratio
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from utils.loss import YoloV3Loss

from torchsummary import summary
# from utils.utils import _evaluate
from utils.datasets import normalize_Tensor

debug_path = "./debug/det_only_code/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

import cv2
import requests
import visdom
import torch.optim as optim

import math

from evaluation.evaluator import Evaluator
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from evaluation.evaluator import validate_with_torchmetrics

def adjust_learning_rate_cosine(batches_done, total_batches=115*200, initial_lr=0.0001, warmup_batches=1000, min_lr_ratio=0.1):
    if batches_done < warmup_batches:
        lr = initial_lr * (batches_done / warmup_batches)
    else:
        progress = (batches_done - warmup_batches) / (total_batches - warmup_batches)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        lr = initial_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
    return lr

def cosine_annealing_with_restarts(batches_done,T_0=2000,T_mult=2,initial_lr=0.0001,min_lr_ratio=0.1,lr_decay_ratio=1.0):
    T_curr = T_0
    total_batches = 0
    cycle = 0  

    while batches_done >= total_batches + T_curr:
        total_batches += T_curr
        T_curr *= T_mult
        cycle += 1

    batch_in_cycle = batches_done - total_batches

    # 사이클마다 최대 lr을 감소시키기 (ex: 0.5면 절반씩 줄어듦)
    current_initial_lr = initial_lr * (lr_decay_ratio ** cycle)

    # cosine annealing 적용
    lr = current_initial_lr * (
        min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * batch_in_cycle / T_curr))
    )

    return lr


def custom_collate_fn(batch):
    imgs, gt, t_p, B_p, boxes, labels = zip(*batch)
    return (
        torch.stack(imgs),
        torch.stack(gt),
        torch.stack(t_p),
        torch.stack(B_p),
        list(boxes),
        list(labels)
    )

def set_seed(seed, torch_set_seed=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if torch_set_seed == True:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_model(model, dataloader, loss_func, args):
    "Compute performance of the model on the validation dataset"
    val_output_dir = args.train_logs + 'val_dir/'
    os.makedirs(val_output_dir, exist_ok=True)
    model.eval()
    val_loss = 0.
    Tensor = torch.cuda.FloatTensor

    # Inference mode
    with torch.inference_mode():
        psnr_val = 0
        for i, batch in enumerate(dataloader):
            # Model inputs
            imgs_distorted = Variable(batch["inp"].type(Tensor))
            imgs_good_gt = Variable(batch["gt"].type(Tensor))
            t_p = Variable(batch["t"].type(Tensor))
            B_p = Variable(batch["B"].type(Tensor))
            # Transpose t_p to 3 channels
            if t_p.shape[1] == 1:
                trans = transforms.Lambda(lambda x: x.repeat(1,3, 1, 1))
                t_p = trans(t_p)

            # Forward pass
            list_J, output_B, output_t, _ = model(imgs_distorted, t_p, B_p)

            output = list_J[-1]
            # Save sample images
            img_sample = torch.cat((imgs_distorted.data, output.data, imgs_good_gt.data), -2)
            save_image(img_sample, val_output_dir +"/%s.jpg" % (i))
            # Compute validation loss
            val_loss += loss_func(output, imgs_good_gt)

            # Compute accuracy and accumulate
            psnr_val += peak_signal_noise_ratio(output, imgs_good_gt)

    return val_loss / len(dataloader.dataset), psnr_val / len(dataloader.dataset)


def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)  # 작은 값 추가해서 NaN 방지

def training(args, vis, train_input_image_window,  train_gt_image_window):

    ## Data pipeline
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    print("Training set loading...")
    train_dataloader = DataLoader(
        Combine_Dataset(args.train_dir, transform=transform),
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4,
        collate_fn = custom_collate_fn,
    )
    
    print('Validation set loading...')
    val_dataloader = DataLoader(
        Combine_Dataset(args.val_dir, transform=transform),
        batch_size=4,
        shuffle=True,
        num_workers=1,
        collate_fn = custom_collate_fn,
    )

    # Detection model load
    det_model = YoloV3(baseline=Darknet53(pretrained=args.det_pretrained), num_classes=args.num_classes)
    det_model.to(DEVICE)
    det_model.anchor.to_device(DEVICE)
    
    print(f"Detection model : {det_model}")
    print(summary(det_model, input_size=(3, args.img_size, args.img_size)))
    rand_img = torch.rand(1, 3, args.img_size, args.img_size).to(DEVICE)
    output = det_model(rand_img)
    print(f"Detection model output : {output[0].shape}, {output[1].shape}, {output[2].shape}")

    params = [p for p in det_model.parameters() if p.requires_grad]
    # print(type(params)) # list
    det_optimizer = optim.Adam(
        params,
        lr=0.0001, 
        weight_decay=0.0005
    )
    # print(f"Learning Rate in Optimizer: {det_optimizer.param_groups[0]['lr']}")
    # print(f"Weight Decay in Optimizer: {det_optimizer.param_groups[0]['weight_decay']}")    

    # Create output directories
    train_output_dir = args.train_logs + 'train_dir/'
    os.makedirs(train_output_dir, exist_ok=True)

    # Create save path
    save_path = args.train_logs + 'save_path/'
    os.makedirs(save_path, exist_ok=True)
    
    # define loss function
    det_criterion = YoloV3Loss(args)
    epoch_0 = 1

    Tensor = torch.cuda.FloatTensor
    best_det_loss = 10
    best_det_AP = 0


    writer = SummaryWriter(comment='BLUE-net', filename_suffix="underwaterimage")
    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    losses_det = []
    losses_lbox = []
    losses_lobj = []
    losses_lcls = []
    lr_list= []
    AP_list =[]
    for epoch in range(epoch_0, args.epochs):
        # model.train()
        det_model.train()  
        # model.zero_grad()
        det_model.zero_grad()

        s = time.time()
        mAP_list =[]
            
        for i, (imgs_distorted, imgs_good_gt, t_p, B_p, boxes, labels) in enumerate(train_dataloader):
            batches_done = len(train_dataloader) * (epoch-1) + i
            imgs_distorted = imgs_distorted.to(DEVICE)  # input image
            imgs_good_gt   = imgs_good_gt.to(DEVICE)    # ground truth
            t_p            = t_p.to(DEVICE)
            B_p            = B_p.to(DEVICE)
            boxes  = [b.to(DEVICE) for b in boxes]
            labels = [l.to(DEVICE) for l in labels]

            # Transpose t_p to 3 channels
            if t_p.shape[1] == 1:
                trans = transforms.Lambda(lambda x: x.repeat(1,3, 1, 1))
                t_p = trans(t_p)
            
            outputs = det_model(imgs_distorted)
            # print(f"Detection model output : {outputs[0].shape}, {outputs[1].shape}, {outputs[2].shape}")

            # det_loss,(loss_xy, loss_wh, loss_obj, loss_noobj, loss_cls) , _ = det_criterion(outputs, boxes, labels, det_model.module.anchor) # DDP로 train 된 경우
            det_loss,(loss_xy, loss_wh, loss_obj, loss_noobj, loss_cls) , _ = det_criterion(outputs, boxes, labels, det_model.anchor) # DDP 인 경우 module 필요
             
            lbox = loss_xy + loss_wh
            lobj = loss_obj
            lcls = loss_cls
            
            e = time.time()
            # total_loss.backward()
            det_optimizer.zero_grad()
            det_loss.backward()
            det_optimizer.step()
            # optimizer.step()

            lr = cosine_annealing_with_restarts(
                batches_done,
                T_0=2000,
                T_mult=1,
                initial_lr=0.0001,
                min_lr_ratio=0.1,
                lr_decay_ratio=0.9  # 🔁 매 cycle마다 절반으로 감소
            )
            for g in det_optimizer.param_groups:
                g['lr'] = lr
            
            if not i % 10:
                    sys.stdout.write(
                        "\r[EPOCH %d / %d][ Batch %d / %d][ Det: %.3f | LBox: %.3f | LObj: %.3f | LCls: %.3f ][ Time: %.3f ]"
                        % (epoch, args.epochs, i, len(train_dataloader),
                        det_loss.item(), lbox.item(), lobj.item(), lcls.item(), (e - s) / 60)
                    )

            writer.add_scalars("Det_loss", {
                "t_det_loss": round(det_loss.item(), 3),
                "t_lbox": round(lbox.item(), 3),
                "t_lobj": round(lobj.item(), 3),
                "t_lcls": round(lcls.item(), 3),
            }, i)

        # writer.add_scalars("Det_loss", {"det_loss": round(det_loss.item(), 3)}, epoch)
        # losses_det.append(round(det_loss.item(), 3))
        writer.add_scalars("Det_loss", {
            "v_det_loss": round(det_loss.item(), 3),
            "v_lbox": round(lbox.item(), 3),
            "v_lobj": round(lobj.item(), 3),
            "v_lcls": round(lcls.item(), 3),
            "v_lr" : det_optimizer.param_groups[0]['lr'],
        }, epoch)
        # print(f"reported v_lr :{det_optimizer.param_groups[0]['lr']}")
        losses_det.append(round(det_loss.item(), 3))
        losses_lbox.append(round(lbox.item(), 3))
        losses_lobj.append(round(lobj.item(), 3))
        losses_lcls.append(round(lcls.item(), 3))
        lr_list.append(det_optimizer.param_groups[0]['lr'])
                  
        mAP50 = validate_with_torchmetrics(det_model, val_dataloader, DEVICE, args)
        print(f"Validation mAP@0.5 = {mAP50:.4f}")
        mAP_list.append(mAP50)
        
        # if metrics_output is not None:
        #     precision, recall, AP, f1, ap_class = metrics_output
        #     evaluation_metrics = [
        #         ("validation/precision", precision.mean()),
        #         ("validation/recall", recall.mean()),
        #         ("validation/mAP", AP.mean()),
        #         ("validation/f1", f1.mean())]
            
        
        # AP_list.append(round(AP.mean().item(), 5))
        # print(f"Det : {mAP}")

        if det_loss < best_det_loss:
            best_det_loss = det_loss
            torch.save(
                {'epoch': epoch, 'model_state_dict': det_model.state_dict(),
                'optimizer_state_dict': det_optimizer.state_dict()},
                os.path.join(save_path, f'best_det_loss_model.pth')
            )
        # if AP.mean() > best_det_AP:
        #     best_det_AP = AP.mean()
        #     torch.save(
        #         {'epoch': epoch, 'model_state_dict': det_model.state_dict(),
        #         'optimizer_state_dict': det_optimizer.state_dict()},
        #         os.path.join(save_path, f'best_AP_model.pth')
        #     )
        if epoch % 10 == 0:
            torch.save(
                {'model_state_dict': det_model.state_dict()},
                os.path.join(save_path, f'{epoch}th_det_model.pth')
            )           

            
    writer.close()


# Function to check if Visdom server is running on a custom port
def check_visdom_server(port=8097):
    url = f'http://localhost:{port}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Visdom server is running on port {port}!")
            return True
        else:
            print(f"Visdom server is not responding on port {port}. Please start the server.")
            return False
    except requests.ConnectionError:
        print(f"Visdom server is not running on port {port}. Please start the server.")
        return False

if __name__ == '__main__':
    # Model-Driven Deep Unfolding Approach to Underwater Image Enhancement
    parser = argparse.ArgumentParser(description='BLUE_Net')
    parser.add_argument('--train_dir', type=str, default='/media/ssd1/hansung/Dataset/0113/train/',help='Training dataset')
    parser.add_argument('--val_dir', type=str, default='/media/ssd1/hansung/Dataset/0113/valid/',help='Validation dataset')
    parser.add_argument('--train_logs', type=str, default='/media/hdd1/hansung/result/BLUE-Net/train_logs/',help='Training logs and outputs')
    parser.add_argument('--seed', type=int, default=3, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',help='initial learning rate (default: 1e-4)')
    parser.add_argument('--n_layers', type=int, default=5, metavar='N',help='number of layers (default: 5)')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from saved checkpoint(s).',)
    parser.add_argument("-m", "--model", type=str, default="/media/hdd1/hansung/BLUENet_Det/config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("--pretrained_weights", type=str, default="weights/darknet53.conv.74")
    parser.add_argument("--det_pretrained", type=str )
    parser.add_argument("--num_classes", type=int, default=4, help="number of classes")
    parser.add_argument("--visdom_port", type=int, default="8099", help="port number for visdom (default: 8098)")
    parser.add_argument("--pred_save_dir", type=str, default='/media/hdd1/hansung/result/BLUE-Net/det_prediction/', help="port number for visdom (default: 8098)")
    parser.add_argument("--class_names", type=list, default=['echinus','holothurian','scallop','starfish'])
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    if not check_visdom_server(port=args.visdom_port):
        print(f"----------------------------------")
        print(f"Exiting... Please start the Visdom server by running: python -m visdom.server -port {args.visdom_port}")
        print(f"Then, run this script again.")
        print(f"----------------------------------")
        exit(0)  # Exit the script if the server is not running+

    run_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    vis = visdom.Visdom(port=args.visdom_port, env=run_version)
    print(f"----------------------------------\n\n")
    print(f" Please open your browser and go to: http://localhost:{args.visdom_port}")
    print(f"\n\n----------------------------------")

    train_input_image_window = vis.image(
        np.random.rand(3, 256, 256),  # Dummy image for initialization
        opts=dict(title='train_input image')
    )

    train_gt_image_window = vis.image(
        np.random.rand(3, 256, 256),  # Dummy image for initialization
        opts=dict(title='train_gt image')
    )
    #run_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.train_logs = f'{args.train_logs}/{run_version}/'
    print(run_version)
    os.makedirs(args.train_logs, exist_ok=True)
    set_seed(args.seed, torch_set_seed=True)
    # Start training
    training(args, vis, train_input_image_window,  train_gt_image_window)