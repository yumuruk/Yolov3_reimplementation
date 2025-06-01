from matplotlib.pylab import f
import torch
import numpy as np
from abc import ABCMeta, abstractmethod


class Anchor(metaclass=ABCMeta):
    def __init__(self, model_name='yolo'):
        self.model_name = model_name.lower()
        assert model_name in ['yolo', 'ssd', 'retina', 'yolov3']

    @abstractmethod
    def create_anchors(self):
        pass


class YOLOv3Anchor(Anchor):
    def __init__(self):
        super().__init__()
        self.anchor_whs = {"small": [(10, 13), (16, 30), (33, 23)],
                           "middle": [(30, 61), (62, 45), (59, 119)],
                           "large": [(116, 90), (156, 198), (373, 326)]}
        self.center_anchors = self.create_anchors()

    def to_device(self, device):
        self.center_anchors[0] = self.center_anchors[0].to(device)
        self.center_anchors[1] = self.center_anchors[1].to(device)
        self.center_anchors[2] = self.center_anchors[2].to(device)

    def anchor_for_scale(self, grid_size, wh):

        center_anchors = []
        for y in range(grid_size):
            for x in range(grid_size):
                cx = x + 0.5
                cy = y + 0.5
                for anchor_wh in wh:
                    w = anchor_wh[0]
                    h = anchor_wh[1]
                    center_anchors.append([cx, cy, w, h])

        print('done!')
        center_anchors_numpy = np.array(center_anchors).astype(np.float32)                          # to numpy  [845, 4]
        center_anchors_tensor = torch.from_numpy(center_anchors_numpy)                              # to tensor [845, 4]
        center_anchors_tensor = center_anchors_tensor.view(grid_size, grid_size, 3, 4)              # [13, 13, 5, 4]
        return center_anchors_tensor

    def create_anchors(self):

        print('make yolo anchor...') ##### 256 ==== URPC dataset image input
        
        actual_strides = {"p_l_stride": 32, # Corresponds to anchor_whs["large"]
                  "p_m_stride": 16, # Corresponds to anchor_whs["middle"]
                  "p_s_stride": 8}  # Corresponds to anchor_whs["small"]

        # 앵커 w,h를 해당 특징맵의 stride로 나누어, 특징맵의 한 칸(grid cell)을 단위로 하는 상대적인 크기로 변환
        # anchor_whs["large"]는 p_l (stride 32)에서 사용
        wh_for_p_l = torch.tensor(self.anchor_whs["large"], dtype=torch.float32) / actual_strides["p_l_stride"]
        # anchor_whs["middle"]는 p_m (stride 16)에서 사용
        wh_for_p_m = torch.tensor(self.anchor_whs["middle"], dtype=torch.float32) / actual_strides["p_m_stride"]
        # anchor_whs["small"]는 p_s (stride 8)에서 사용
        wh_for_p_s = torch.tensor(self.anchor_whs["small"], dtype=torch.float32) / actual_strides["p_s_stride"]

        # 각 특징 맵의 그리드 크기 (입력 크기가 256일 때)
        grid_size_p_l = 256 // actual_strides["p_l_stride"]  # 8x8
        grid_size_p_m = 256 // actual_strides["p_m_stride"]  # 16x16
        grid_size_p_s = 256 // actual_strides["p_s_stride"]  # 32x32

        # anchor_for_scale 함수는 이제 스케일링된 wh 값을 받음
        # 모델 출력 [p_l, p_m, p_s] 순서에 맞춰서 생성 및 저장
        center_anchors_p_l = self.anchor_for_scale(grid_size_p_l, wh_for_p_l)
        center_anchors_p_m = self.anchor_for_scale(grid_size_p_m, wh_for_p_m)
        center_anchors_p_s = self.anchor_for_scale(grid_size_p_s, wh_for_p_s)

        # self.center_anchors 에 저장되는 순서가 중요
        # YoloV3.predict는 [anchor_for_p_l, anchor_for_p_m, anchor_for_p_s] 순서를 가정
        return [center_anchors_p_l, center_anchors_p_m, center_anchors_p_s]

