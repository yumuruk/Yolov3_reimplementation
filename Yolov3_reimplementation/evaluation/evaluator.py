import os
import json
import tempfile

from matplotlib.pylab import f
from evaluation.voc_eval import voc_eval
from pycocotools.cocoeval import COCOeval
import torch
from utils.utils import cxcy_to_xy, xywh2xyxy
from tqdm import tqdm

from torchmetrics.detection.mean_ap import MeanAveragePrecision


class Evaluator(object):
    def __init__(self, opts):
        self.opts = opts
        self.data_type = opts.data_type

        # for VOC
        self.det_img_name = list()
        self.det_additional = list()
        self.det_boxes = list()
        self.det_labels = list()
        self.det_scores = list()

        # for COCO
        self.results = list()
        self.img_ids = list()
        
        # for custom
        self.metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=False)
        self.preds  = []
        self.targets = []

    def get_info(self, info):
        if self.data_type == 'voc':

            (pred_boxes, pred_labels, pred_scores, img_names, additional_info) = info

            self.det_img_name.append(img_names)  # 4952 len list # [1] - img_name_length [B, k]
            self.det_additional.append(additional_info)  # 4952 len list # [2] -  w, h   [B, 2]
            self.det_boxes.append(pred_boxes)  # 4952 len list # [obj, 4]
            self.det_labels.append(pred_labels)  # 4952 len list # [obj]
            self.det_scores.append(pred_scores)  # 4952 len list # [obj]

        elif self.data_type == 'coco':

            (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids) = info

            self.img_ids.append(img_id)

            # convert coco_results coordination
            pred_boxes[:, 2] -= pred_boxes[:, 0]  # x2 to w
            pred_boxes[:, 3] -= pred_boxes[:, 1]  # y2 to h

            w = img_info['width']
            h = img_info['height']

            pred_boxes[:, 0] *= w
            pred_boxes[:, 2] *= w
            pred_boxes[:, 1] *= h
            pred_boxes[:, 3] *= h

            for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                if int(pred_label) == 80:  # background label is 80
                    print('background label :', int(pred_label))
                    continue

                coco_result = {
                    'image_id': img_id,
                    'category_id': coco_ids[int(pred_label)],
                    'score': float(pred_score),
                    'bbox': pred_box.tolist(),
                }
                self.results.append(coco_result)
                
        elif self.data_type == 'custom':
            # info: (pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels) 
            pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels = info
            # torchmetrics 형식에 맞춰 dict 생성
            self.preds.append({
                'boxes':  pred_boxes.cpu(),
                'scores': pred_scores.cpu(),
                'labels': pred_labels.cpu(),
            })
            self.targets.append({
                'boxes':  gt_boxes.cpu(),
                'labels': gt_labels.cpu(),
            })
        return
            
            

    def evaluate(self, dataset):
        if self.data_type == 'voc':

            test_root = os.path.join(dataset.root, 'VOCtest_06-Nov-2007', 'VOCdevkit', 'VOC2007', 'Annotations')
            mAP = voc_eval(opts=self.opts,
                           test_xml_path=test_root,
                           img_names=self.det_img_name,
                           additional=self.det_additional,
                           bboxes=self.det_boxes,
                           scores=self.det_scores,
                           classes=self.det_labels)

        elif self.data_type == 'coco':

            _, tmp = tempfile.mkstemp()
            json.dump(self.results, open(tmp, "w"))

            cocoGt = dataset.coco
            cocoDt = cocoGt.loadRes(tmp)

            # https://github.com/argusswift/YOLOv4-pytorch/blob/master/eval/cocoapi_evaluator.py
            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.

            coco_eval = COCOeval(cocoGt=cocoGt, cocoDt=cocoDt, iouType='bbox')
            coco_eval.params.imgIds = self.img_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            mAP = coco_eval.stats[0]
            mAP_50 = coco_eval.stats[1]
        return mAP
    
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

def validate_with_torchmetrics(det_model, dataloader, device, args):
    """
    Compute mAP@0.5 on a custom dataset by running YOLOv3.predict per image.
    GT boxes are assumed to be in (cx, cy, w, h) format and 0-1 normalized.
    """
    det_model.to(device).eval()
    if hasattr(det_model, 'anchor') and hasattr(det_model.anchor, 'to_device'):
        det_model.anchor.to_device(device)
    
    # box_format='xyxy'가 기본값이지만 명시적으로 표시 가능
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=False, box_format='xyxy').to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"): # tqdm 추가
            # custom_collate_fn 반환 순서: imgs, gt, t_p, B_p, boxes, labels
            # 여기서 gt, t_p, B_p는 사용하지 않음
            input_images, _, _, _, gt_boxes_batch_list, gt_labels_batch_list = batch
            B = input_images.size(0)

            imgs = input_images.to(device)

            # 모델의 predict 메소드가 사용할 앵커 준비
            # YOLOv3Anchor 클래스의 self.center_anchors가 [anchor_for_p_l, anchor_for_p_m, anchor_for_p_s] 순서라고 가정
            # 그리고 YoloV3.predict가 이 순서를 기대한다고 가정
            center_anchors_for_predict = det_model.anchor.center_anchors

            # 배치 내 각 이미지에 대해 predict 호출 및 metric 업데이트
            preds_for_metric_update = []
            targets_for_metric_update = []

            # 모델의 raw 출력을 먼저 얻음 (배치 단위)
            # preds_l, preds_m, preds_s 각각 [B, grid, grid, channels]
            raw_preds_l, raw_preds_m, raw_preds_s = det_model(imgs)

            for i in range(B):
                # i번째 이미지에 대한 raw 예측 슬라이싱
                single_raw_preds = (
                    raw_preds_l[i].unsqueeze(0), # 배치 차원 유지 (1, grid, grid, channels)
                    raw_preds_m[i].unsqueeze(0),
                    raw_preds_s[i].unsqueeze(0),
                )

                # det_model.predict는 (pred_boxes_xyxy, pred_labels, pred_scores)를 NumPy 배열로 반환
                # opts/args 인자는 현재 YoloV3.predict 내부의 _suppress에서 하드코딩되어 사용되지 않음.
                # 필요하다면 YoloV3._suppress 수정 또는 args에 conf_thres, top_k 등 추가.
                np_pred_boxes_xyxy, np_pred_labels, np_pred_scores = det_model.predict(
                    single_raw_preds, center_anchors_for_predict, args # args 전달 (내부에서 사용 안될 수 있음)
                )

                # NumPy 배열을 Tensor로 변환
                pred_boxes_tensor = torch.tensor(np_pred_boxes_xyxy,  device=device, dtype=torch.float32)
                pred_labels_tensor = torch.tensor(np_pred_labels, device=device, dtype=torch.long)
                pred_scores_tensor = torch.tensor(np_pred_scores, device=device, dtype=torch.float32)

                preds_for_metric_update.append({
                    'boxes': pred_boxes_tensor,
                    'scores': pred_scores_tensor,
                    'labels': pred_labels_tensor
                })

                # i번째 이미지에 대한 GT 가져오기
                # gt_boxes_batch_list[i]는 (num_obj, 4) 크기의 (cx, cy, w, h) 0-1 정규화 텐서
                gt_boxes_cxcywh_current = gt_boxes_batch_list[i].to(device)
                gt_labels_current = gt_labels_batch_list[i].to(device)

                if gt_boxes_cxcywh_current.numel() > 0:
                    # GT 박스 변환: (cx, cy, w, h) -> (xmin, ymin, xmax, ymax)
                    gt_boxes_xyxy_current = xywh2xyxy(gt_boxes_cxcywh_current)
                    # 좌표값이 0~1 범위를 벗어나지 않도록 clamp
                    # gt_boxes_xyxy_current = torch.clamp(gt_boxes_xyxy_current, 0, 1)
                else:
                    gt_boxes_xyxy_current = torch.empty((0, 4), device=device, dtype=torch.float32)
                
                targets_for_metric_update.append({
                    'boxes': gt_boxes_xyxy_current,
                    'labels': gt_labels_current
                })
            
            # 배치 전체에 대한 예측과 타겟으로 metric 업데이트
            if preds_for_metric_update and targets_for_metric_update:
                metric.update(preds_for_metric_update, targets_for_metric_update)

    # 모든 배치 처리 후 최종 mAP 계산
    computed_map = metric.compute()
    map_50 = computed_map['map_50'].item() # .item()으로 스칼라 값 추출
    
    # train_only_det.py의 print문과 일치시키기 위해 map_50만 반환
    return map_50