from __future__ import division
# py libs
import torch
import torch.nn as nn
from net import *
from torchsummary import summary

import sys
import os
from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchvision.utils import save_image
from PIL import Image

def cxcy_to_xy(cxcy):

    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=-1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
debug_path = "./debug/original_code/"


###########################################################################
######################## enhancement model ################################
###########################################################################

# Minimum color loss principle
def get_mean_value(batch):
    # Get batch size of input
    batch_size = batch.shape[0]
    # Create output
    list_mean_sorted = []
    list_indices = []
    largest_index = []
    medium_index= []
    smallest_index = []
    largest_channel = []
    medium_channel = []
    smallest_channel = []

    # Get the largest,  medium, and smallest value/ channels.
    for bs in range(batch_size):
        image = batch[bs,:,:,:]
        mean = torch.mean(image, (2,1))
        mean_I_sorted, indices = torch.sort(mean)
        list_mean_sorted.append(mean_I_sorted)
        list_indices.append(indices)
        # Index of largest, medium and smallest value.
        largest_index.append(indices[2])
        medium_index.append(indices[1])
        smallest_index.append(indices[0])
        # Get largest, medium and smallest channel
        largest_channel.append(torch.unsqueeze(image[indices[2],:,:], 0))
        medium_channel.append(torch.unsqueeze(image[indices[1],:,:], 0))
        smallest_channel.append(torch.unsqueeze(image[indices[0],:,:], 0))

    # Sort list mean values
    list_mean_sorted = torch.stack(list_mean_sorted)
    list_indices = torch.stack(list_indices)
    # Get final index
    largest_index = torch.stack(largest_index)
    medium_index = torch.stack(medium_index)
    smallest_index = torch.stack(smallest_index)
    # Get final channel
    largest_channel = torch.stack(largest_channel)
    medium_channel = torch.stack(medium_channel)
    smallest_channel = torch.stack(smallest_channel)

    return list_mean_sorted, list_indices, largest_channel, medium_channel, smallest_channel, largest_index, medium_index, smallest_index

def mapping_index(batch, value, index):
    # Mapping the index to channel
    batch_size = batch.shape[0]
    new_batch = []
    for bs in range(batch_size):
        image = batch[bs,:,:,:]
        image[index[bs],:,:] = value[bs]
        new_batch.append(image)
    new_batch = torch.stack(new_batch)
    return new_batch
    
# Get the dark channel of an image (B, C, H, W) -> (B, 1, H, W)
def get_dark_channel(x, patch_size):
    pad_size = (patch_size - 1) // 2
    # Get batch size of input
    H, W = x.size()[2], x.size()[3]
    # Minimum among three channels
    x, _ = x.min(dim=1, keepdim=True)  # (B, 1, H, W)
    x = nn.ReflectionPad2d(pad_size)(x)  # (B, 1, H+2p, W+2p)
    x = nn.Unfold(patch_size)(x) # (B, k*k, H*W)
    x = x.unsqueeze(1)  # (B, 1, k*k, H*W)
    
    # Minimum in (k, k) patch
    index_map = torch.argmin(x, dim=2, keepdim=False)
    dark_map, _ = x.min(dim=2, keepdim=False)  # (B, 1, H*W)
    dark_map = dark_map.view(-1, 1, H, W)

    return dark_map, index_map

# Soft thresholding function for tensor x and threshold lamda 
def softThresh(x, lamda):
    relu = nn.ReLU()
    return torch.sign(x).cuda() * relu(torch.abs(x).cuda() - lamda)

# Proposed algorithm
class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        print('Loading subnetworks .....')
        H_Net = [RDN(3,1)]
        self.H_Net = nn.Sequential(*H_Net)

        self.t_1D_Net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=False),
            # nn.ReLU()
        )
        # Learnable parameter
        self.gamma_1 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.gamma_2 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.gamma_3 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.gamma_4 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.gamma_5 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.eta_1 = nn.Parameter(torch.tensor([1.001]),requires_grad=True)
        self.eta_2 = nn.Parameter(torch.tensor([1.001]),requires_grad=True)

    def forward(self, I, t_p, B_p, B, t, J, G, H, P, Q, u, v, X, Y, patch_size = 35, eps = 1e-6):
        lambda_1 = 1.0
        lambda_2 = 0.7
        lambda_3 = 0.3
        lambda_4 = 1.0
        lambda_5 = 1.0
        gamma_1 = self.gamma_1
        gamma_2 = self.gamma_2
        gamma_3 = self.gamma_3
        gamma_4 = self.gamma_4
        gamma_5 = self.gamma_5
        # eta_1 = self.eta_1
        # eta_2 = self.eta_2
        ## Minimum color loss principle 
        # (argmin )┬(J_m,J_s )⁡〖‖(J_l ) ̅-(J_m ) ̅ ‖_1+‖(J_l ) ̅-(J_s ) ̅ ‖_1 〗
        # Calculate J_l, J_m, J_s
        list_mean_sorted, list_indices, J_l, J_m, J_s, largest_index, medium_index, smallest_index = get_mean_value(J)
        J_l_bar = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(list_mean_sorted[:,2],1),1),1).to(DEVICE)
        J_m_bar = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(list_mean_sorted[:,1],1),1),1).to(DEVICE)
        J_s_bar = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(list_mean_sorted[:,0],1),1),1).to(DEVICE)

        J_l = J_l.to(DEVICE)
        J_m = J_m.to(DEVICE)
        J_s = J_s.to(DEVICE)

        J_m = J_m + torch.mul(J_l_bar - J_m_bar, J_l)
        J_s = J_s + torch.mul(J_l_bar - J_s_bar, J_l)

        J = mapping_index(J.clone(), J_m.clone(), medium_index)
        J = mapping_index(J.clone(), J_s.clone(), smallest_index)

        # Math modules
        ## B-module
        D = torch.ones(I.shape).to(DEVICE)
        # print(((gamma_1)*B_p))
        B = (lambda_3*B_p - lambda_1*(J*t - I)*(1 - t))/(lambda_1*(1.0 - t)*(1 - t) + lambda_3)
        B = torch.mean(B,(2,3), True)
        # B = self.B_Net(B)
        B = B*D

        ## t-module
        t = (lambda_2*t_p + gamma_4*H - lambda_1*(B - I)*(J - B) - X)/(lambda_1*(J - B)*(J - B) + lambda_2 + gamma_4)
        # t = torch.mean(t, 1, keepdim=True)
        t = self.t_1D_Net(t)
        t = torch.cat((t,t,t), 1)

        M_T_P = u
        M_T_Q = v

        ## J-module
        J = (lambda_1*(t*(I - B*(1.0 - t))) + gamma_3*G + gamma_4*u - gamma_5*v - Y + gamma_5)/(lambda_1*t*t + gamma_3 + gamma_4 + gamma_5)
        # J = (lambda_1*(t*(I - B*(1.0 - t))) + gamma_3*G + gamma_4*u - Y)/(lambda_1*t*t + gamma_3 + gamma_4)
        # J = (beta*Y - Q - (B*(1.0 - t) - I)*t)/(t*t + beta)

        u = (gamma_1*M_T_P + gamma_4*J)/(gamma_1 + gamma_4)
        v = (gamma_2*M_T_Q - gamma_5*J + gamma_5)/(gamma_2 + gamma_5)

        ## Z_Net
        H = self.H_Net(t + (1.0/gamma_4) * X)

        ## P & Q module
        X = X + gamma_4*(t - H)
        Y = Y + gamma_3*(J - G)

        M_u, index_map_dark = get_dark_channel(u, patch_size)
        M_v, index_map_dark = get_dark_channel(v, patch_size)

        ## M & N module
        P = softThresh(M_u, lambda_4/gamma_1)
        Q = softThresh(M_v, lambda_5/gamma_1)
        # Q = P
        return B, t, J, G, H, P, Q, u, v, X, Y, gamma_3

class IPMM(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(IPMM, self).__init__()
        act = nn.PReLU()
        self.shallow_feat2 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.merge12=mergeblock(n_feat,3,True)
    
    def forward(self,x2_img, stage1_img,feat1,res1,x2_samfeats):
        ## PMM
        x2 = self.shallow_feat2(x2_img)
        x2_cat = self.merge12(x2, x2_samfeats)
        feat2,feat_fin2 = self.stage2_encoder(x2_cat, feat1, res1)
        res2 = self.stage2_decoder(feat_fin2,feat2)
        x3_samfeats, stage2_img = self.sam23(res2[-1], x2_img)
        return x3_samfeats, stage2_img, feat2, res2

class Dual_Net(torch.nn.Module):
    def __init__(self, LayerNo):
        super(Dual_Net, self).__init__()

        self.LayerNo = LayerNo
        net_layers = []
        for i in range(LayerNo):
            net_layers.append(BasicBlock())
        self.uunet = nn.ModuleList(net_layers)
        # DGUNet
        in_c=3
        out_c=3
        n_feat=40
        scale_unetfeats=20
        scale_orsnetfeats=16
        num_cab=8
        kernel_size=3
        reduction=4
        bias=False
        depth=5
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4)
        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.merge12=mergeblock(n_feat,3,True)
        self.basic = IPMM(in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False)

    def forward(self,I, t_p, B_p):

        bs, _, _, _ = I.shape
        B = torch.zeros((bs,3,1,1)).to(DEVICE)
        t = torch.zeros(I.shape).to(DEVICE)
        J = I.to(DEVICE)

        G = torch.zeros(I.shape).to(DEVICE)
        H = torch.zeros(I.shape).to(DEVICE)
        X = torch.zeros(I.shape).to(DEVICE)
        Y = torch.zeros(I.shape).to(DEVICE)

        P = torch.zeros(I.shape).to(DEVICE)
        Q = torch.zeros(I.shape).to(DEVICE)

        u = torch.zeros(I.shape).to(DEVICE)
        v = torch.zeros(I.shape).to(DEVICE)

        list_J = []
        list_B = []
        list_t = []
        # list_G = []
        # list_H = []
        # list_u = []
        # list_v = []
        # list_P = []
        # list_Q = []

        # IPMM Module
        # 1st stage Proximal
        gamma_3 = torch.tensor([3.001]).to(DEVICE)
        x1_img = J + (1.0/gamma_3) * Y
        x1 = self.shallow_feat1(x1_img)
        feat1,feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1,feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)

        G = stage1_img

        for j in range(self.LayerNo):

            [B, t, J, G, H, P, Q, u, v, X, Y, gamma_3] = self.uunet[j](I, t_p, B_p, B, t, J, G, H, P, Q, u, v, X, Y)
            
            # IPMM Module
            img = J + (1.0/gamma_3) * Y
            x2_samfeats, stage1_img, feat1, res1 = self.basic(img, stage1_img,feat1,res1,x2_samfeats)
            G = stage1_img
            list_J.append(J)
            list_B.append(B)
            list_t.append(t)
            # list_G.append(G)
            # list_H.append(torch.cat((H,H,H), 1))
            # list_u.append(u)
            # list_v.append(v)
            # list_P.append(torch.cat((P,P,P), 1))
            # list_Q.append(torch.cat((Q,Q,Q), 1))

        # return list_J, list_B, list_t, list_G, list_H, list_u, list_v, list_P, list_Q
        return list_J, list_B, list_t
    
###########################################################################
######################## detection model ################################
###########################################################################

import os
import wget
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.anchors import YOLOv3Anchor
from utils.utils import bar_custom, cxcy_to_xy
from torchvision.ops.boxes import nms as torchvision_nms


class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channel, in_channel//2, 1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(in_channel//2, momentum=0.9),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(in_channel//2, in_channel, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(in_channel, momentum=0.9),
                                      nn.LeakyReLU(0.1),
                                      )

    def forward(self, x):
        residual = x
        x = self.features(x)
        x += residual
        return x


class Darknet53(nn.Module):

    '''
    num_params
    pytorch
    40584928
    
    darknet
    40620640
    '''

    def __init__(self, block=ResBlock, num_classes=1000, pretrained=True):
        super().__init__()

        self.num_classes = num_classes
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(0.1),
            self.make_layer(block, in_channels=64, num_blocks=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.1),
            self.make_layer(block, in_channels=128, num_blocks=2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.1),
            self.make_layer(block, in_channels=256, num_blocks=8),
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.LeakyReLU(0.1),
            self.make_layer(block, in_channels=512, num_blocks=8),
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024, momentum=0.9),
            nn.LeakyReLU(0.1),
            self.make_layer(block, in_channels=1024, num_blocks=4),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)
        self.init_layer()

        if pretrained:
            self.darknet53_url = "https://pjreddie.com/media/files/darknet53.conv.74"
            file = "darknet53.pth"
            directory = os.path.join(torch.hub.get_dir(), 'checkpoints')
            os.makedirs(directory, exist_ok=True)
            pretrained_file = os.path.join(directory, file)
            if os.path.exists(pretrained_file):
                print("weight already exist...!")
            else:
                print("Download darknet53 pre-trained weight...")
                wget.download(url=self.darknet53_url, out=pretrained_file, bar=bar_custom)
                print('')
                print("Done...!")
            self.load_darknet_weights(pretrained_file)


    def init_layer(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.gap(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        conv_layer = None
        # refer to https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py
        for i, module in enumerate(self.modules()):
            if isinstance(module, nn.Conv2d):
                conv_layer = module
            if isinstance(module, nn.BatchNorm2d):
                bn_layer = module
                num_b = bn_layer.bias.numel()  # Number of biases

                # Bias
                bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b

                # Weight
                bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b

                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b

                # Running Var
                bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                continue
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


class YoloV3(nn.Module):
    def __init__(self, baseline, num_classes=80):
        super().__init__()
        self.baseline = baseline
        self.anchor = YOLOv3Anchor()
        self.anchor_number = 3
        self.num_classes = num_classes

        self.extra_conv1 = nn.Sequential(
            nn.Conv2d(384, 128, 1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 128, 1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 128, 1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pred_1 = nn.Conv2d(256, self.anchor_number * (1 + 4 + self.num_classes), 1)

        self.extra_conv2 = nn.Sequential(
            nn.Conv2d(768, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pred_2 = nn.Conv2d(512, self.anchor_number * (1 + 4 + self.num_classes), 1)
        self.conv_128x1x1 = nn.Sequential(
            nn.Conv2d(512, 128, 1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.extra_conv3 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pred_3 = nn.Conv2d(1024, self.anchor_number * (1 + 4 + self.num_classes), 1)
        self.conv_256x1x1 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.initialize()
        print("num_params : ", self.count_parameters())

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        self.extra_conv1.apply(init_layer)
        self.extra_conv2.apply(init_layer)
        self.extra_conv3.apply(init_layer)
        self.pred_1.apply(init_layer)
        self.pred_2.apply(init_layer)
        self.pred_3.apply(init_layer)
        self.conv_128x1x1.apply(init_layer)
        self.conv_256x1x1.apply(init_layer)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        p1 = x = self.baseline.features1(x)  # after residual3 [B, 256, 52, 52]
        p2 = x = self.baseline.features2(x)  # after residual4 [B, 512, 26, 26]
        p3 = x = self.baseline.features3(x)  # after residual5 [B, 1024, 13, 13]

        p3_up = self.conv_256x1x1(p3)                        # [B, 256, 13, 13]
        p3_up = F.interpolate(p3_up, scale_factor=2.)        # [B, 256, 26, 26]
        p3 = self.extra_conv3(p3)                            # [B, 1024, 13, 13]
        p3 = self.pred_3(p3)                                 # [B, 255, 13, 13] ***

        p2 = torch.cat([p2, p3_up], dim=1)                   # [B, 768, 26, 26]
        p2 = self.extra_conv2(p2)                            # [B, 512, 26, 26]

        p2_up = self.conv_128x1x1(p2)                        # [B, 128, 26, 26]
        p2_up = F.interpolate(p2_up, scale_factor=2.)        # [B, 128, 52, 52]
        p2 = self.pred_2(p2)                                 # [B, 255, 26, 26] ***

        p1 = torch.cat([p1, p2_up], dim=1)                   # [B, 384, 52, 52]
        p1 = self.extra_conv1(p1)                            # [B, 256, 52, 52]
        p1 = self.pred_1(p1)                                 # [B, 255, 52, 52] ***

        p_s = p1.permute(0, 2, 3, 1)  # B, 52, 52, 255
        p_m = p2.permute(0, 2, 3, 1)  # B, 26, 26, 255
        p_l = p3.permute(0, 2, 3, 1)  # B, 13, 13, 255

        return [p_l, p_m, p_s]

    def pred2target(self, pred):
        # pred to target
        out_size = pred.size(1)  # 13, 13
        pred_targets = pred.view(-1, out_size, out_size, 3, 5 + self.num_classes)
        pred_target_xy = pred_targets[..., :2].sigmoid()  # 0, 1 sigmoid(tx, ty) -> bx, by
        pred_target_wh = pred_targets[..., 2:4]  # 2, 3
        pred_objectness = pred_targets[..., 4].unsqueeze(-1).sigmoid()  # 4        class probability
        pred_classes = pred_targets[..., 5:].sigmoid()  # 20 / 80  classes
        return pred_target_xy, pred_target_wh, pred_objectness, pred_classes

    def decodeYoloV3(self, pred_targets, center_anchor):

        # pred to target
        out_size = pred_targets.size(1)  # 13, 13
        pred_txty, pred_twth, pred_objectness, pred_classes = self.pred2target(pred_targets)

        # decode
        center_anchors_xy = center_anchor[..., :2]  # torch.Size([13, 13, 3, 2])
        center_anchors_wh = center_anchor[..., 2:]  # torch.Size([13, 13, 3, 2])

        pred_bbox_xy = center_anchors_xy.floor().expand_as(pred_txty) + pred_txty
        pred_bbox_wh = center_anchors_wh.expand_as(pred_twth) * pred_twth.exp()
        pred_bbox = torch.cat([pred_bbox_xy, pred_bbox_wh], dim=-1)                     # [B, 13, 13, 3, 4]
        pred_bbox = pred_bbox.view(-1, out_size * out_size * 3, 4) / out_size           # [B, 507, 4]
        pred_cls = pred_classes.reshape(-1, out_size * out_size * 3, self.num_classes)  # [B, 507, 80]
        pred_conf = pred_objectness.reshape(-1, out_size * out_size * 3)
        return pred_bbox, pred_cls, pred_conf

    def predict(self, preds, list_center_anchors, opts):
        pred_targets_l, pred_targets_m, pred_targets_s = preds
        center_anchor_l, center_anchor_m, center_anchor_s = list_center_anchors

        pred_bbox_l, pred_cls_l, pred_conf_l = self.decodeYoloV3(pred_targets_l, center_anchor_l)
        pred_bbox_m, pred_cls_m, pred_conf_m = self.decodeYoloV3(pred_targets_m, center_anchor_m)
        pred_bbox_s, pred_cls_s, pred_conf_s = self.decodeYoloV3(pred_targets_s, center_anchor_s)

        pred_bbox = torch.cat([pred_bbox_l, pred_bbox_m, pred_bbox_s], dim=1)                # [B, 10647, 4]
        pred_cls = torch.cat([pred_cls_l, pred_cls_m, pred_cls_s], dim=1)                    # [B, 10647, 80]
        pred_conf = torch.cat([pred_conf_l, pred_conf_m, pred_conf_s], dim=1)

        scores = (pred_cls * pred_conf.unsqueeze(-1))          # [B, 10647, 21]
        boxes = cxcy_to_xy(pred_bbox).clamp(0, 1)              # [B, 10647, 4]

        bbox, label, score = self._suppress(boxes, scores, opts)
        return bbox, label, score

    def _suppress(self, raw_cls_bbox, raw_prob, opts):

        raw_cls_bbox = raw_cls_bbox.cpu()
        raw_prob = raw_prob.cpu()

        bbox = list()
        label = list()
        score = list()

        for l in range(1, self.num_classes):
            prob_l = raw_prob[..., l]
            ## 내가 수정한 부분
            # mask = prob_l > opts.conf_thres
            mask = prob_l > 0.5
            cls_bbox_l = raw_cls_bbox[mask]
            prob_l = prob_l[mask]
            keep = torchvision_nms(cls_bbox_l, prob_l, iou_threshold=0.45)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l - 1) * np.ones(len(keep)))
            score.append(prob_l[keep].cpu().numpy())

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        n_objects = score.shape[0]
        # top_k = opts.top_k
        ## 내가 수정한 부분
        ## 한 이미지당 최대 객체 수를 50으로 제한
        top_k = 50
        # if n_objects > opts.top_k:
        if n_objects > top_k:
            sort_ind = score.argsort(axis=0)[::-1]  # [::-1] means descending
            score = score[sort_ind][:top_k]  # (top_k)
            bbox = bbox[sort_ind][:top_k]  # (top_k, 4)
            label = label[sort_ind][:top_k]  # (top_k)
        return bbox, label, score


if __name__ == '__main__':
    img = torch.randn([1, 3, 256, 256])
    model = YoloV3(Darknet53(pretrained=True))
    p_l, p_m, p_s = model(img)

    print("large : ", p_l.size())
    print("medium : ", p_m.size())
    print("small : ", p_s.size())

    '''
    torch.Size([1, 52, 52, 255])
    torch.Size([1, 26, 26, 255])
    torch.Size([1, 13, 13, 255])
    '''

