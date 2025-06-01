import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def compute_mean_std(image_folder):
    transform = transforms.ToTensor()  # 이미지를 [0,1] 범위의 tensor로 변환
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_pixels = 0

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in tqdm(image_files, desc="Calculating mean and std"):
        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)

        num_pixels += img_tensor.shape[1] * img_tensor.shape[2]  # H * W
        mean += img_tensor.sum(dim=[1, 2])
        std += (img_tensor ** 2).sum(dim=[1, 2])

    mean /= num_pixels
    std = (std / num_pixels - mean ** 2).sqrt()

    return mean, std

# 사용 예시
dataset_path = "/media/hdd/hansung/0113/train/gt/"  # 너의 이미지 폴더 경로
mean, std = compute_mean_std(dataset_path)
print("Mean:", mean)
print("Std:", std)
