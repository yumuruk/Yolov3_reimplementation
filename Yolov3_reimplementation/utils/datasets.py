from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision.transforms.functional import normalize

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)  # 작은 값 추가해서 NaN 방지

def normalize_with_mean_std(tensor):
    # 전체 데이터셋 기준 정규화 값
    
    # for enhanced urpc with bluenet
    mean = torch.tensor([0.6211, 0.6103, 0.5884], dtype=tensor.dtype)
    std = torch.tensor([0.1775, 0.1972, 0.2015], dtype=tensor.dtype)

    # mean = torch.tensor([0.2390, 0.5614, 0.3069], dtype=tensor.dtype)
    # std = torch.tensor([0.0483, 0.1072, 0.0688], dtype=tensor.dtype)

    # Device 및 shape 맞추기 (Batch x C x H x W 기준)
    mean = mean.to(tensor.device).view(1, 3, 1, 1)
    std = std.to(tensor.device).view(1, 3, 1, 1)

    return (tensor - mean) / std

def normalize_Tensor(tensor):
    
    # for enhanced urpc with bluenet
    mean = [0.6211, 0.6103, 0.5884]
    std  = [0.1775, 0.1972, 0.2015]
    
    # for original dataset
    # mean = [0.2390, 0.5614, 0.3069]
    # std = [0.0483, 0.1072, 0.0688]
    
    normalized_tensor = normalize(tensor, mean, std)
    return normalized_tensor

def unnormalize_Tensor(tensor):
    # same mean/std as normalize_Tensor
    mean = torch.tensor([0.6211, 0.6103, 0.5884], dtype=tensor.dtype, device=tensor.device)
    std  = torch.tensor([0.1775, 0.1972, 0.2015], dtype=tensor.dtype, device=tensor.device)
    # (C,) → (1,C,1,1) 혹은 (B,C,1,1) 형태로 브로드캐스트
    shape = [1] * tensor.dim()
    shape[1] = 3
    mean = mean.view(shape)
    std  = std.view(shape)
    # inverse normalize
    return tensor * std + mean


def min_max_normalize_batch(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    각 이미지(B 기준) 별로 min-max 정규화를 벡터화 연산으로 수행.
    입력: (B, C, H, W) 텐서
    출력: 동일 shape, 정규화된 텐서
    """
    # (B, 1, 1, 1) 형태로 이미지별 최소/최대값 추출
    min_vals = batch_tensor.amin(dim=(1, 2, 3), keepdim=True)
    max_vals = batch_tensor.amax(dim=(1, 2, 3), keepdim=True)

    # 정규화
    normalized = (batch_tensor - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)
