"""
    Modules for processing training/ testing data.
"""
import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
# import albumentations as A
import torch
from pathlib import Path
from typing import Optional
 
class UIEBDataset(Dataset):
    """ Common data pipeline to organize and generate
         training pairs for various datasets   
    """
    def __init__(self, root, img_size=256, transform=None, is_test = False):
        self.input_files, self.gt_files, self.t_p_files, self.B_p_files, self.label_files = self.get_file_paths(root, is_test)
        # print(len(self.input_files), len(self.gt_files), len(self.t_p_files), len(self.B_p_files), len(self.label_files))
        # print(f'input :{len(self.input_files)}, gt :  {len(self.gt_files)}, t_p : {len(self.t_p_files)}, label : {len(self.label_files)}')
        self.len = min(len(self.input_files), len(self.gt_files), len(self.t_p_files), len(self.label_files))
        
        self.img_size = img_size
        self.max_objects = 100
        # self.multiscale = multiscale
        # self.min_size = self.img_size - 3 * 32
        # self.max_size = self.img_size + 3 * 32
        # self.batch_count = 0
        self.transform = transform
        
        if transform is not None:
            self.transform = T.Compose(transform)
        else:
            # No legacy augmentations
            # Paper uses flipping and rotation transform to obtain 7 augmented versions of data
            # Rotate by 90, 180, 270 degs, hflip, vflip? Not very clear
            # This is as close as it gets without having to go out of my way to reproduce exactly 7 augmented versions
            self.transform = T.Compose(
                [   T.Resize((self.img_size, self.img_size)),
                    T.ToTensor(),]
            )
        
        

    def __getitem__(self, index):
        
        input_image = Image.open(self.input_files[index % self.len])
        gt_image = Image.open(self.gt_files[index % self.len])
        t_p = Image.open(self.t_p_files[index % self.len])
        B_p = Image.open(self.B_p_files[index % self.len])
        
        input_image = self.transform(input_image)
        gt_image = self.transform(gt_image)
        t_p = self.transform(t_p)
        B_p = self.transform(B_p)
        # 파일 경로에서 파일명만 추출
        file_names = os.path.basename(self.input_files[index % self.len])  # 파일명만 추출
        
        
        # 라벨 파일 불러오기
        label_file_path = self.label_files[index % self.len]
        with open(label_file_path, 'r') as f:
            lines = f.readlines()  # 모든 줄 읽기

            # 각 줄을 공백 기준으로 분리하고 float 형태로 변환
            labels = []  # 결과를 저장할 리스트

            for line in lines:
                if line.strip():  # 빈 줄을 무시하고 처리
                    float_values = [float(value) for value in line.strip().split()]
                    labels.append(float_values)
                
            if not labels:
                # print(f'{file_names} 이미지 라벨 없음 !')
                labels_tensor = torch.zeros((0, 5), dtype=torch.float32)  # YOLO 방식 (라벨이 없을 경우 빈 텐서)
            else:
                # 최대 길이 계산
                # max_length = max(len(label) for label in labels)

                # NaN을 추가해 각 배열의 길이를 최대 길이에 맞추고 2D NumPy 배열로 변환
                # padded_labels = np.array([np.pad(label, (0, max_length - len(label)), constant_values=np.nan) for label in labels])

                # NumPy 배열을 텐서로 변환
                labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # labels = torch.zeros(labels_tensor.shape[0], 6) # (num_samples, max_length, 1)
        # labels[:,1:] = labels_tensor
        # input_filename = os.path.basename(input_image)
                
        labels = torch.zeros(labels_tensor.shape[0], 6)  # (num_samples, max_length, 1)
        labels[:, 1:] = labels_tensor

        
        # return input_image, gt_image, t_p, B_p, labels
        return input_image, gt_image, t_p, B_p, labels, file_names
    
    def __len__(self):
        return self.len 

    def get_file_paths(self, root, is_test):
        if is_test == True:
            input_files = sorted(glob.glob(os.path.join(root, 'images') + "/*.*"))
            t_p_files = sorted(glob.glob(os.path.join(root, '416_t_prior') + "/*.*"))
            B_p_files = sorted(glob.glob(os.path.join(root, '416_B_prior') + "/*.*"))
            label_files = sorted(glob.glob(os.path.join(root, 'labels') + "/*.*"))
            gt_files = []
        else:
            input_files = sorted(glob.glob(os.path.join(root, 'images') + "/*.*"))
            gt_files = sorted(glob.glob(os.path.join(root, 'gt') + "/*.*"))
            t_p_files = sorted(glob.glob(os.path.join(root, '416_t_prior') + "/*.*"))
            B_p_files = sorted(glob.glob(os.path.join(root, '416_B_prior') + "/*.*"))
            label_files = sorted(glob.glob(os.path.join(root, 'labels') + "/*.*"))
            
        return input_files, gt_files, t_p_files, B_p_files, label_files
    
    
class non_GTDataset(Dataset):
    """ Common data pipeline to organize and generate
         training pairs for various datasets   
    """
    def __init__(self, root, img_size=256, transform=None, is_test = False):
        self.input_files,  self.label_files = self.get_file_paths(root, is_test)
        # print(f'input :{len(self.input_files)}, gt :  {len(self.gt_files)}, t_p : {len(self.t_p_files)}, label : {len(self.label_files)}')
        self.len = min(len(self.input_files), len(self.label_files))
        
        self.img_size = img_size
        self.max_objects = 100
        # self.multiscale = multiscale
        # self.min_size = self.img_size - 3 * 32
        # self.max_size = self.img_size + 3 * 32
        # self.batch_count = 0
        self.transform = transform
        
        if transform is not None:
            self.transform = T.Compose(transform)
        else:
            # No legacy augmentations
            # Paper uses flipping and rotation transform to obtain 7 augmented versions of data
            # Rotate by 90, 180, 270 degs, hflip, vflip? Not very clear
            # This is as close as it gets without having to go out of my way to reproduce exactly 7 augmented versions
            self.transform = T.Compose(
                [   T.Resize((self.img_size, self.img_size)),
                    T.ToTensor(),]
            )
        
        

    def __getitem__(self, index):
        
        input_image = Image.open(self.input_files[index % self.len])     
        input_image = self.transform(input_image)
        
        # 파일 경로에서 파일명만 추출
        file_names = os.path.basename(self.input_files[index % self.len])  # 파일명만 추출
        
        # 라벨 파일 불러오기
        label_file_path = self.label_files[index % self.len]
        with open(label_file_path, 'r') as f:
            lines = f.readlines()  # 모든 줄 읽기

            # 각 줄을 공백 기준으로 분리하고 float 형태로 변환
            labels = []  # 결과를 저장할 리스트

            for line in lines:
                if line.strip():  # 빈 줄을 무시하고 처리
                    float_values = [float(value) for value in line.strip().split()]
                    labels.append(float_values)
                
            if not labels:
                # print(f'{file_names} 이미지 라벨 없음 !')
                labels_tensor = torch.zeros((0, 5), dtype=torch.float32)  # YOLO 방식 (라벨이 없을 경우 빈 텐서)
            else:
                # 최대 길이 계산
                # max_length = max(len(label) for label in labels)

                # NaN을 추가해 각 배열의 길이를 최대 길이에 맞추고 2D NumPy 배열로 변환
                # padded_labels = np.array([np.pad(label, (0, max_length - len(label)), constant_values=np.nan) for label in labels])

                # NumPy 배열을 텐서로 변환
                labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # labels = torch.zeros(labels_tensor.shape[0], 6) # (num_samples, max_length, 1)
        # labels[:,1:] = labels_tensor
        # input_filename = os.path.basename(input_image)
                
        labels = torch.zeros(labels_tensor.shape[0], 6)  # (num_samples, max_length, 1)
        labels[:, 1:] = labels_tensor

        
        # return input_image, gt_image, t_p, B_p, labels
        return input_image, labels, file_names
    
    def __len__(self):
        return self.len 

    def get_file_paths(self, root, is_test):

        if is_test == True:
            input_files = sorted(glob.glob(os.path.join(root, 'images') + "/*.*"))
            label_files = sorted(glob.glob(os.path.join(root, 'labels') + "/*.*"))
        else:
            input_files = sorted(glob.glob(os.path.join(root, 'images') + "/*.*"))
            label_files = sorted(glob.glob(os.path.join(root, 'labels') + "/*.*"))
        return input_files, label_files

class Combine_Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform or T.Compose([
            T.Resize((416, 416)),   # enhancement/detection 모두를 위한 크기
            T.ToTensor()
        ])
        self.input_files, self.gt_files, self.t_p_files, self.B_p_files, self.label_files = self.get_file_paths(root)
        self.len = len(self.input_files)

    def get_file_paths(self, root):
        input_files = sorted(glob.glob(os.path.join(root, 'images', '*.*')))
        gt_files = sorted(glob.glob(os.path.join(root, 'gt', '*.*')))
        t_p_files = sorted(glob.glob(os.path.join(root, '416_t_prior', '*.*')))
        B_p_files = sorted(glob.glob(os.path.join(root, '416_B_prior', '*.*')))
        label_files = sorted(glob.glob(os.path.join(root, 'labels', '*.*')))
        return input_files, gt_files, t_p_files, B_p_files, label_files

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # --- 이미지 불러오기 및 변환 ---
        input_image = Image.open(self.input_files[idx]).convert("RGB")
        gt_image = Image.open(self.gt_files[idx]).convert("RGB")
        t_p_image = Image.open(self.t_p_files[idx]).convert("L")  # t_p는 1채널
        B_p_image = Image.open(self.B_p_files[idx]).convert("RGB")

        input_image = self.transform(input_image)
        gt_image = self.transform(gt_image)
        t_p_image = self.transform(t_p_image)  # 1채널 그대로
        B_p_image = self.transform(B_p_image)

        # --- 라벨 로딩 및 변환 ---
        boxes = []
        class_ids = []
        label_path = self.label_files[idx]
        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) == 5:
                    class_id, x, y, w, h = parts
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(int(class_id))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_ids = torch.zeros((0,),   dtype=torch.long)
        else:
            boxes     = torch.tensor(boxes,     dtype=torch.float32)
            class_ids = torch.tensor(class_ids, dtype=torch.long)

        return input_image, gt_image, t_p_image, B_p_image, boxes, class_ids