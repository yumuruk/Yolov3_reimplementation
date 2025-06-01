from cProfile import label
import sys

from matplotlib.pylab import f
from matplotlib.pyplot import step
from data_utils import UIEBDataset
import os
import torch
import argparse
from model import Dual_Net, load_model
from utils.loss import compute_loss
from datetime import datetime
import time
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.functional import peak_signal_noise_ratio
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from utils.datasets import normalize_tensor, normalize_with_mean_std, min_max_normalize_batch
from utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info

from torchsummary import summary
from utils.utils import _evaluate, combine_evaluate

debug_path = "./debug/original_code/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

import cv2
import requests
import visdom
import torch.optim as optim
import math

def cosine_annealing_with_restarts(batches_done,T_0=2000,T_mult=2,initial_lr=0.0001,min_lr_ratio=0.1,lr_decay_ratio=1.0):
    T_curr = T_0
    total_batches = 0
    cycle = 0  # 현재 재시작 사이클 횟수

    # 현재 사이클 위치 계산
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
    # input_images, gt_image, t_p, B_p, labels = zip(*batch)
    input_images, gt_image, t_p, B_p, labels, file_names = zip(*batch)
    input_images = torch.stack(input_images)
    gt_images = torch.stack(gt_image)
    t_p = torch.stack(t_p)
    B_p = torch.stack(B_p)

    # Add sample index to targets
    for i, boxes in enumerate(labels):
        boxes[:, 0] = i
    labels = torch.cat(labels, 0)

    return {"inp": input_images,
        "gt": gt_images,
        "t": t_p,
        "B": B_p,
        "labels": labels,
        "file_names": file_names  # 추가된 부분
    }



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
    """
    Compute average loss and PSNR over the *entire* validation dataset.
    """
    val_output_dir = args.train_logs + 'val_dir/'
    os.makedirs(val_output_dir, exist_ok=True)
    model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    total_loss = 0.0
    total_psnr = 0.0
    total_samples = 0

    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            # Model inputs
            imgs_distorted = batch["inp"].type(Tensor).to(DEVICE)
            imgs_good_gt   = batch["gt"].type(Tensor).to(DEVICE)
            t_p            = batch["t"].type(Tensor).to(DEVICE)
            B_p            = batch["B"].type(Tensor).to(DEVICE)
            filename = batch["file_names"]

            # (optional) transpose t_p to 3 channels if needed
            if t_p.shape[1] == 1:
                t_p = t_p.repeat(1, 3, 1, 1)

            # Forward
            output_J, output_B, output_t = model(imgs_distorted, t_p, B_p)
            fin_en_img = output_J[-1]  # Final enhanced image
            # for k in range(len(output_J)):
            #     output[k] = min_max_normalize_batch(output[k])
            #     # output[k] = normalize_with_mean_std(output[k])
            #     output = det_model(output[k])
            #     loss = det_loss (output)
            #     #########33
            #     # weighted sumn
            #     ########
            #     loss_list.append(loss.item())

            
            # total_loss = loss_list[]
            # Save a few samples for inspection
            if i < 5:  # 예시로 처음 5배치만
                img_sample = torch.cat((imgs_distorted, fin_en_img, imgs_good_gt), dim=-2)
                save_image(img_sample, os.path.join(val_output_dir, f"{i}.jpg"))

            # 배치 단위 loss와 PSNR (각각 batch 평균)
            batch_size = imgs_distorted.size(0)
            loss_batch = loss_func(fin_en_img, imgs_good_gt)               # 평균 L1 loss over this batch
            psnr_batch = peak_signal_noise_ratio(fin_en_img, imgs_good_gt) # 평균 PSNR over this batch

            # 샘플 수 고려해 ‘합’으로 누적
            total_loss   += loss_batch.item() * batch_size
            total_psnr   += psnr_batch.item() * batch_size
            total_samples += batch_size

    # 전체 샘플 수로 나눠서 진짜 sample‐wise 평균 계산
    avg_loss = total_loss / total_samples
    avg_psnr = total_psnr / total_samples

    return avg_loss, avg_psnr



def training(args, vis, train_input_image_window,train_output_image_window, train_gt_image_window):

    ## Data pipeline
    transform = [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ]

    print("Training set loading...")
    train_dataloader = DataLoader(
        UIEBDataset(args.train_dir, transform=transform),
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4,
        collate_fn = custom_collate_fn,
    )

    print('Validation set loading...')
    val_dataloader = DataLoader(
        UIEBDataset(args.val_dir, transform=transform),
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn = custom_collate_fn,
    )

    # Define loss function for training
    loss_1 = torch.nn.L1Loss()

    # Enhancement model load
    n_layer = args.n_layers
    model = Dual_Net(LayerNo = n_layer)
    # model = nn.DataParallel(model)
    
    if args.en_pretrained_weights is not None:
        checkpoint = torch.load(args.en_pretrained_weights)
        state_dict = checkpoint['model_state_dict']
        state_dict = {
            (k[7:] if k.startswith('module.') else k): v 
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict) 
        model.to(DEVICE)
    else:
        model.to(DEVICE)

    # Detection model load
    det_model = load_model(args.model, args.pretrained_weights) ## args.model이랑 pretrained weights가져와야함
    # det_model = nn.DataParallel(det_model)
    det_model.to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (0.9, 0.99))

    # Detection Optimizer
    params = [p for p in det_model.parameters() if p.requires_grad]
    det_optimizer = optim.Adam(
        params,
        lr=0.00005, 
        weight_decay=0.00005
    )

    # Create output directories
    train_output_dir = args.train_logs + 'train_dir/'
    os.makedirs(train_output_dir, exist_ok=True)

    # Create save path
    save_path = args.train_logs + 'save_path/'
    os.makedirs(save_path, exist_ok=True)        
    epoch_0 = 1

    Tensor = torch.cuda.FloatTensor
    ## Training pipeline
    best_acc = 0
    best_det_loss = 1
    best_det_AP = 0
    
    writer = SummaryWriter(comment='BLUE-net', filename_suffix="underwaterimage")
    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    val_loss_list =[]
    val_psnr_list = []
    val_AP_list = []
    det_lr_list = []
    
    total_loss = torch.tensor(0.0, device=DEVICE)
    for epoch in range(epoch_0, args.epochs):
        model.train(); det_model.train() 
        
        # model.zero_grad()
        # det_model.zero_grad()
        
        labels = []
        s = time.time()
    
        for i, batch in enumerate(train_dataloader):
            # Zero gradients
            optimizer.zero_grad()
            det_optimizer.zero_grad()
            
            batches_done = len(train_dataloader) * epoch + i
            imgs_distorted = batch["inp"].type(Tensor)
            imgs_good_gt = batch["gt"].type(Tensor)
            t_p = batch["t"].type(Tensor)
            B_p = batch["B"].type(Tensor)
            labels_tensor = batch["labels"].to(DEVICE)
            file_name = batch["file_names"]
            
            # print(f'input :{len(imgs_distorted)}, gt :  {len(imgs_good_gt)}, t_p : {len(t_p)}, B_p : {len(B_p)}, label : {len(labels_tensor)}')
            # print(f'input :{imgs_distorted.shape}, gt :  {imgs_good_gt.shape}, t_p : {t_p.shape}, B_p : {B_p.shape}, label : {labels_tensor.shape}')
            
            # Transpose t_p to 3 channels
            if t_p.shape[1] == 1:
                trans = transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
                t_p = trans(t_p)

            # Forward pass
            total_loss = 0.0
            output_J, output_B, output_t = model(imgs_distorted, t_p, B_p)    
            rec_loss_list, det_loss_list = [], []
            # print(f"output J length : {len(output_J)}") ## 5가 나와야함
            for k, img_out in enumerate(output_J):
                # reconstruction
                in_rec_loss = loss_1(img_out, imgs_good_gt)
                rec_loss_list.append(2*(k+1)*in_rec_loss)
                
                # normalize image before detection and comparison                
                save_image(img_out, os.path.join(train_output_dir, f"{file_name[0]}_output_J_{k}.jpg"))
                img_out = min_max_normalize_batch(img_out)
                save_image(img_out, os.path.join(train_output_dir, f"{file_name[0]}_output_J_{k}_norm.jpg"))
                
                # detection
                det_out = det_model(img_out)
                in_det_loss, _ = compute_loss(det_out, labels_tensor, det_model)
                det_loss_list.append(2*(k+1)*in_det_loss)

                # iteration 별 로스 로깅 (선택)
                writer.add_scalars(f"Loss/iter_{k}", {
                    "rec": in_rec_loss.item(),
                    "det": in_det_loss.item(),
                }, global_step=epoch * len(train_dataloader) + i)

            # 2) 합산된 loss 계산
            sum_rec_loss = (torch.stack(rec_loss_list).sum())/(len(output_J)*(len(output_J)+1))
            sum_det_loss = (torch.stack(det_loss_list).sum())/(len(output_J)*(len(output_J)+1))

            # 3) adaptive weight
            # weight = 1.0 - (epoch / (args.epochs - 1))
            

            # visulization_path = "./loss_debug/"
            # save_image(output_J[-1], os.path.join(visulization_path, f"{file_name[0]}_output_J_final.jpg"))
            # save_image(imgs_good_gt, os.path.join(visulization_path, f"{file_name[0]}_gt.jpg"))
            # save_image(imgs_distorted, os.path.join(visulization_path, f"{file_name[0]}_input.jpg"))
            l1_out = torch.abs(imgs_good_gt - output_J[-1]).mean()
            l1_dist = torch.abs(imgs_good_gt - imgs_distorted).mean()
            # print(f"l1_out : {l1_out.item()}, l1_dist : {l1_dist.item()}")
            weight = (l1_out / (l1_dist)).clamp(0.0, 1.0)
            # print(f"weight : {weight.item()}")
            #### 여기 부분 체크중

            # 4) 최종 loss
            total_loss = (1-weight) * sum_rec_loss + weight * sum_det_loss

            # 5) batch 단위로 요약 로깅
            writer.add_scalars("Loss/summary", {
                "sum_rec": sum_rec_loss.item(),
                "sum_det": sum_det_loss.item(),
                "total": total_loss.item(),
                "weight": weight.item(),
            }, global_step=epoch * len(train_dataloader) + i)
                
            with torch.no_grad():
                vis.images(imgs_distorted[0,:,:,:].cpu(), win=train_input_image_window, opts=dict(title="train_input image"))
                vis.images(imgs_good_gt[0,:,:,:].cpu(), win=train_gt_image_window, opts=dict(title="train_gt image"))
                # vis.images(normalize_tensor(output_G[-1]).cpu(), win=train_G_image_window, opts=dict(title="train_output_G image"))
                # vis.images(normalize_tensor(enhanced[0,:,:,:]).cpu(), win=train_output_image_window, opts=dict(title="train_output_J image"))

                # ## list of batch images for enhanced images
                # save_image(enhanced, os.path.join(debug_path, "after_J.png"))
                # save_image(output_B[-1], os.path.join(debug_path, "output_B.png"))
                # save_image(output_t[-1], os.path.join(debug_path, "output_T.png"))

            e = time.time()
            total_loss.backward()
            optimizer.step()

            lr = cosine_annealing_with_restarts(    
                batches_done,
                T_0=100000,
                T_mult=1,
                initial_lr=0.0001,
                min_lr_ratio=0.1,
                lr_decay_ratio=0.6  
            )
            det_lr_list.append(lr)
            for g in det_optimizer.param_groups:
                g['lr'] = lr
            det_optimizer.step()
            
            with torch.no_grad() :
                ## Print log
                if not i%10:
                    sys.stdout.write("\r[EPOCH%d/%d:batch%d/%d][Loss:%.3f][Rec:%.3f][Det:%.3f][Weight:%.4f][Time:%.3f]"
                                    %(epoch, args.epochs, i, len(train_dataloader),sum_rec_loss.item(), total_loss.item(), sum_det_loss.item(),weight, (e-s)/60)
                    )
                writer.add_scalars("Train_loss", {"train_total_loss": round(total_loss.item(), 3)}, i)
                writer.add_scalars("Rec_loss", {"recon_loss": round((weight * sum_rec_loss).item(), 3)}, i)
                writer.add_scalars("Det_loss", {"det_loss": ((1 - weight) * sum_det_loss)}, i)
                writer.add_scalar("Learning_rate", lr, i)
                vis.line(X=np.array([i]), Y=np.array([total_loss.item()]), win='train_loss', update='append', opts=dict(title='Training Loss (per step)', xlabel='Step', ylabel='Loss'))
                vis.line(X=np.array([i]), Y=np.array([sum_rec_loss.item()]), win='train_loss_rec', update='append', opts=dict(title='Reconstruction Loss (per step)', xlabel='Step', ylabel='Loss'))
                vis.line(X=np.array([i]), Y=np.array([sum_det_loss.item()]), win='train_loss_det', update='append', opts=dict(title='Detection Loss (per step)', xlabel='Step', ylabel='Loss'))
                
        with torch.no_grad():

            # Validate enhancement model
            val_loss, val_psnr = validate_model(model, val_dataloader, loss_1, args)
            
            # Validate detection model
            metrics_output = combine_evaluate(
                det_model,
                model,
                val_dataloader,
                args.class_names,
                img_size=args.img_size,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                verbose=True,
            )
            
            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]

            # Print log
            # print(f"Train Loss: {total_loss.item():.3f}, Valid Loss: {val_loss.item():3f}, PSNR: {val_psnr.item():.2f} Det_loss : {det_loss.item():.3f}, AP : {AP.mean().item():.4f}")
            print(f"Train Loss: {total_loss.item():.3f}, Valid Loss: {val_loss:.3f}, PSNR: {val_psnr:.2f},  AP : {AP.mean():.4f}")
            val_loss_list.append(round(val_loss, 3))
            val_psnr_list.append(round(val_psnr, 2))
            val_AP_list.append(round(AP.mean(),4))

            # Validation loss 기록
            writer.add_scalar('val/val_loss', val_loss, epoch)
            writer.add_scalar('val/val_psnr', val_psnr, epoch)
            writer.add_scalar('val/val_AP', AP.mean().item(), epoch)
            writer.add_scalar('train/epoch', epoch + 1, epoch)

            ## Save model checkpoints
            if val_psnr > best_acc:
                best_acc = val_psnr
                torch.save(
                    {'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    os.path.join(save_path, f'model.pth')
                )
            if AP.mean() > best_det_AP:
                best_det_AP = AP.mean()
                torch.save(
                    {'epoch': epoch, 'model_state_dict': det_model.state_dict(),
                    'optimizer_state_dict': det_optimizer.state_dict()},
                    os.path.join(save_path, f'best_AP_model.pth')
                )
            
            if epoch % 40 == 0:
                torch.save(
                    {
                        'model_state_dict': det_model.state_dict(),
                        'learning_rate': det_optimizer.param_groups[0]['lr']
                    },
                    os.path.join(save_path, f'{epoch}th_det_model.pth')
                )            
            vis.line(X=np.arange(1, len(val_loss_list) + 1), Y=np.array(val_loss_list), win='val_loss', opts=dict(title='Validation Loss',       xlabel='Epoch', ylabel='Loss'))
            vis.line(X=np.arange(1, len(val_psnr_list) + 1), Y=np.array(val_psnr_list), win='val_psnr', opts=dict(title='Validation PSNR',     xlabel='Epoch', ylabel='PSNR'))
            vis.line(X=np.arange(1, len(val_AP_list) + 1),   Y=np.array(val_AP_list),   win='val_AP',     opts=dict(title='Validation AP',       xlabel='Epoch', ylabel='AP'))
            vis.line(X=np.arange(1, len(det_lr_list) + 1),   Y=np.array(det_lr_list),   win='det_lr',     opts=dict(title='Detection Learning Rate', xlabel='Step', ylabel='LR'))          
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
    # parser.add_argument('--train_dir', type=str, default='/media/ssd/hansung/Dataset/URPC_enhanced/BLUE/train/',help='Training dataset')
    # parser.add_argument('--val_dir', type=str, default='/media/ssd/hansung/Dataset/URPC_enhanced/BLUE/valid/',help='Validation dataset')
    parser.add_argument('--train_dir', type=str, default='/media/ssd1/hansung/Dataset/0113/train/',help='Training dataset')
    parser.add_argument('--val_dir', type=str, default='/media/ssd1/hansung/Dataset/0113/valid/',help='Validation dataset')
    # parser.add_argument('--train_dir', type=str, default='/media/ssd/hansung/Dataset/0113_test/train/',help='Training dataset')
    # parser.add_argument('--val_dir', type=str, default='/media/ssd/hansung/Dataset/0113_test/valid/',help='Validation dataset')
    # parser.add_argument('--train_dir', type=str, default='/media/ssd/hansung/Dataset/0113_debug/train/',help='Training dataset')
    # parser.add_argument('--val_dir', type=str, default='/media/ssd/hansung/Dataset/0113_debug/valid/',help='Validation dataset')
    parser.add_argument('--train_logs', type=str, default='/media/hdd1/hansung/result/BLUE-Net/train_logs/',help='Training logs and outputs')
    parser.add_argument('--seed', type=int, default=5, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='input batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',help='initial learning rate (default: 1e-4)')
    parser.add_argument('--n_layers', type=int, default=5, metavar='N',help='number of layers (default: 5)')
    parser.add_argument('--en_pretrained_weights', type=str, help='pth file for pretrained enhancement network.',)
    parser.add_argument("--pretrained_weights", type=str, default ="/media/hdd1/hansung/BLUENet_Det_0404/weights/darknet53.conv.74", help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    # parser.add_argument('--en_pretrained_weights', type=str, default="/media/hdd/hansung/result/BLUE-Net/train_logs/20250418_085609/save_path/model.pth", help='pth file for pretrained enhancement network.')
    # parser.add_argument("--pretrained_weights", type=str, default ="/media/hdd/hansung/result/BLUE-Net/train_logs/20250419_011839/save_path/best_AP_model.pth", help="Path to checkpoint file (.weights or .pth). Starts training from chesckpoint model")
    parser.add_argument('--resume', type=str, default=None, help='Resume training from saved checkpoint(s).',)
    parser.add_argument("-m", "--model", type=str, default="/media/hdd1/hansung/BLUENet_Det/config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("--visdom_port", type=int, default="8099", help="port number for visdom (default: 8098)")
    parser.add_argument("--pred_save_dir", type=str, default='/media/hdd1/hansung/result/BLUE-Net/det_prediction/', help="port number for visdom (default: 8098)")
    parser.add_argument("--class_names", type=list, default=['echinus','holothurian','scallop','starfish'])
    parser.add_argument("--img_size", type=int, default=256 )
    
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

    train_output_image_window = vis.image(
        np.random.rand(3, 256, 256),  # Dummy image for initialization
        opts=dict(title='train_output image')
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
    training(args, vis, train_input_image_window,train_output_image_window, train_gt_image_window)
    