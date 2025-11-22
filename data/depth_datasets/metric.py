# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import cv2
import time
from scipy.optimize import linear_sum_assignment
from PIL import Image


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()
        # 检查是否在分布式环境中
        self.distributed = dist.is_available() and dist.is_initialized()
        # 获取当前进程的rank
        self.rank = dist.get_rank() if self.distributed else 0

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def all_gather(self):
        """聚合所有进程的指标数据，仅主进程返回聚合结果"""
        if not self.distributed:
            return self.result()

        # 准备要发送的数据
        keys = list(self._data.index)
        local_totals = [self._data.loc[key, "total"] for key in keys]
        local_counts = [self._data.loc[key, "counts"] for key in keys]

        # 将数据转换为张量以便分布式通信
        totals_tensor = torch.tensor(
            local_totals, dtype=torch.float64, device=self._get_device()
        )
        counts_tensor = torch.tensor(
            local_counts, dtype=torch.float64, device=self._get_device()
        )

        # 收集所有进程的数据
        gathered_totals = [
            torch.zeros_like(totals_tensor) for _ in range(dist.get_world_size())
        ]
        gathered_counts = [
            torch.zeros_like(counts_tensor) for _ in range(dist.get_world_size())
        ]

        dist.all_gather(gathered_totals, totals_tensor)
        dist.all_gather(gathered_counts, counts_tensor)

        # 主进程计算全局平均值
        if self.rank == 0:
            global_totals = torch.sum(torch.stack(gathered_totals), dim=0)
            global_counts = torch.sum(torch.stack(gathered_counts), dim=0)

            # 计算全局平均
            global_averages = {}
            for i, key in enumerate(keys):
                if global_counts[i] > 0:
                    global_averages[key] = (global_totals[i] / global_counts[i]).item()
                else:
                    global_averages[key] = 0.0
            return global_averages
        else:
            return None

    def _get_device(self):
        """获取当前进程使用的设备"""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.rank}")
        return torch.device("cpu")


# -------------------- Depth Metrics --------------------


def delete(points):
    points = np.asarray(points)
    goal = []
    for i in range(points.shape[0] - 1):
        a = abs(points[0][0] - points[i + 1][0])
        b = abs(points[0][1] - points[i + 1][1])
        if a > 5 or b > 5:
            goal.append(points[i + 1])
    goal.append(points[0])
    if len(goal) != points.shape[0]:
        goal = delete(goal)
    return goal


def cnt_area(cnt):
    """
    sorting contours by contour area size
    """
    area = cv2.contourArea(cnt)

    return area


def extract_bbox(pred_img, gt_img):
    """
    input one image
    return coodinates of all bboxes in this image
    """
    start = time.time()
    bgr_img = np.array(pred_img)
    h, w, c = np.array(gt_img).shape
    bgr_img = cv2.resize(bgr_img, (w, h))
    output = bgr_img
    bgr_img = cv2.medianBlur(bgr_img, 3)  # median filtering
    bgr_img = cv2.bilateralFilter(bgr_img, 0, 0, 30)  # bilateral filtering
    output_img = bgr_img

    # extracting blue areas using hsv channels
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # low_hsv = np.array([156, 43, 46])
    # high_hsv = np.array([180, 255, 255])
    # mask1 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    # low_hsv = np.array([0, 43, 46])
    # high_hsv = np.array([10, 255, 255])
    low_hsv = np.array([100, 130, 50])
    high_hsv = np.array([125, 255, 255])
    mask1 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

    # extracting the red area of the original image
    for i in np.arange(0, bgr_img.shape[0], 1):
        for j in np.arange(0, bgr_img.shape[1], 1):
            # if mask1[i, j] == 0 and mask2[i, j] == 0:
            if mask1[i, j] == 0:
                output_img[i, j, :] = 0

    # get grayscale map
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # threshold segmentation based on grayscale
    otsuThe, dst_Otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dst_Otsu = cv2.Canny(dst_Otsu, 50, 150, apertureSize=3)

    # image binarization
    ret, binary = cv2.threshold(
        dst_Otsu, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE
    )

    # find the outline
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = list(contours)
    contours.sort(key=cnt_area, reverse=False)

    bboxes = []

    # contour judgment
    for obj in contours:  # objs in one image
        bbox = []

        area = cv2.contourArea(obj)  # calculate the area of the area within the contour
        perimeter = cv2.arcLength(obj, True)  # calculate the contour perimeter
        approx = cv2.approxPolyDP(
            obj, 0.02 * perimeter, True
        )  # get the coordinates of the contour corner points
        x, y, w, h = cv2.boundingRect(
            approx
        )  # get coordinate values and width and height

        if perimeter < 10:  # remove small contour areas
            for i in np.arange(x, x + w, 1):
                for j in np.arange(y, y + h, 1):
                    binary[j, i] = 0
        else:
            cv2.rectangle(
                output, (x, y), (x + w, y + h), (0, 255, 255), 1
            )  # draw the bounding box
            bbox = np.zeros((4,))  # print border coordinates

            bbox[0] = x
            bbox[1] = y
            bbox[2] = x + w
            bbox[3] = y + h
            bboxes.append(list(bbox))

    end = time.time()
    # print("one image done, cost time:{}".format(end - start))
    return bboxes, output


def ext_coor(pred_img, gt_img, cls_name):
    img_name = cls_name
    img = np.array(pred_img)
    h, w, c = np.array(gt_img).shape
    img = cv2.resize(img, (w, h))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # low_hsv = np.array([156, 43, 46])
    # high_hsv = np.array([180, 255, 255])
    low_hsv = np.array([100, 110, 70])
    high_hsv = np.array([140, 255, 255])
    mask1 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    # low_hsv = np.array([0, 43, 46])
    # high_hsv = np.array([10, 255, 255])
    # mask2 = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    # mask = cv2.add(mask2, mask1)
    mask = mask1

    kernel = np.ones((3, 3), "uint8")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.copyMakeBorder(mask, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)

    points = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (
                np.sum(np.array(mask[i + 50, j + 50 : j + 70])) == 255 * 20
                and np.sum(np.array(mask[i + 50 : i + 70, j + 50])) == 255 * 20
                and np.sum(np.array(mask[i + 50 : i + 60, j + 40 : j + 50])) < 255 * 50
                and np.sum(np.array(mask[i + 40 : i + 50, j + 50 : j + 60])) < 255 * 50
                and np.sum(np.array(mask[i + 40 : i + 50, j + 40 : j + 50])) < 255 * 50
                and np.sum(np.array(mask[i + 50 : i + 60, j + 50 : j + 60])) < 255 * 50
            ):

                cv2.circle(img, (j, i), 1, (0, 255, 0), -1)
                points.append([i, j])
            if (
                np.sum(np.array(mask[i + 50, j + 30 : j + 50])) == 255 * 20
                and np.sum(np.array(mask[i + 30 : i + 50, j + 50])) == 255 * 20
                and np.sum(np.array(mask[i + 40 : i + 50, j + 50 : j + 60])) < 255 * 50
                and np.sum(np.array(mask[i + 50 : i + 60, j + 40 : j + 50])) < 255 * 50
                and np.sum(np.array(mask[i + 50 : i + 60, j + 50 : j + 60])) < 255 * 50
                and np.sum(np.array(mask[i + 40 : i + 50, j + 40 : j + 50])) < 255 * 50
            ):
                cv2.circle(img, (j, i), 2, (0, 255, 0), -1)
                points.append([i, j])

    points = np.array(points)
    if points.size == 0:
        return [], img

    points_sort = pd.DataFrame(points, columns=["x", "y"])
    points_sort.sort_values(by=["x", "y"], axis=0)

    goal = delete(points)
    goal = pd.DataFrame(goal, columns=["x", "y"])
    goal = goal.sort_values(by=["x", "y"], axis=0)
    goal = np.array(goal)
    point = []
    for i in range(goal.shape[0]):
        for j in np.arange(i + 1, goal.shape[0]):
            point.append([goal[i, 0], goal[i, 1], goal[j, 0], goal[j, 1]])
    point_new = []
    for i in range(len(point)):
        if point[i][1] < point[i][3]:
            point_new.append(point[i])

    img_vis = img.copy()

    if len(point_new) == 0:
        return [], img

    bboxes = []
    for i in range(len(point_new)):
        xx1 = point_new[i][1]
        yy1 = point_new[i][0]
        xx2 = point_new[i][3]
        yy2 = point_new[i][2]
        cv2.rectangle(img_vis, (xx1, yy1), (xx2, yy2), (0, 255, 0), 2)

        bbox = [int(xx1), int(yy1), int(xx2), int(yy2)]
        bboxes.append(bbox)
        cv2.putText(
            img_vis,
            img_name,
            (xx1 + 10, yy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
    return bboxes, img_vis


def _iou_matrix(gt_bboxes, pred_bboxes, pixel_inclusive=False, eps=1e-6):
    """Compute IoU matrix between GT (Gx4) and Pred (Px4)."""
    if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
        return np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=np.float32)

    gt = np.asarray(gt_bboxes, dtype=np.float32)
    pr = np.asarray(pred_bboxes, dtype=np.float32)

    # gt: [G,4], pr: [P,4]
    gx1, gy1, gx2, gy2 = gt[:, 0:1], gt[:, 1:2], gt[:, 2:3], gt[:, 3:4]  # [G,1]
    px1, py1, px2, py2 = (
        pr[None, :, 0],
        pr[None, :, 1],
        pr[None, :, 2],
        pr[None, :, 3],
    )  # [1,P]

    xx1 = np.maximum(gx1, px1)  # [G,P]
    yy1 = np.maximum(gy1, py1)
    xx2 = np.minimum(gx2, px2)
    yy2 = np.minimum(gy2, py2)

    offs = 1.0 if pixel_inclusive else 0.0
    iw = np.maximum(0.0, xx2 - xx1 + offs)
    ih = np.maximum(0.0, yy2 - yy1 + offs)
    inter = iw * ih

    ga = np.maximum(0.0, (gx2 - gx1 + offs)) * np.maximum(
        0.0, (gy2 - gy1 + offs)
    )  # [G,1]
    pa = np.maximum(0.0, (px2 - px1 + offs)) * np.maximum(
        0.0, (py2 - py1 + offs)
    )  # [1,P]
    union = ga + pa - inter + eps

    return (inter / union).astype(np.float32)  # [G,P]


def cal_bboxes_iou(gt_bboxes, pred_bboxes, iou_threshold=0.5, pixel_inclusive=False):
    """
    使用匈牙利算法将 GT 与 Pred 一对一匹配。
    返回：
      - pred_for_each_gt: 长度 == len(gt_bboxes)，按 GT 顺序对齐的预测框或 None（未匹配）
      - matches: [(gi, pj, iou), ...] 仅包含 IoU>=阈值 的匹配对
      - unmatched_gt:  未匹配的 GT 索引列表
      - unmatched_pred: 未匹配的 Pred 索引列表
    """
    G, P = len(gt_bboxes), len(pred_bboxes)
    # 边界情况
    if G == 0:
        return [], [], [], list(range(P))
    if P == 0:
        return [None] * G, [], list(range(G)), []

    ious = _iou_matrix(gt_bboxes, pred_bboxes, pixel_inclusive=pixel_inclusive)  # [G,P]
    cost = 1.0 - ious  # 最小化代价 == 最大化 IoU

    # 匈牙利：返回行索引(对应GT)与列索引(对应Pred)
    row_ind, col_ind = linear_sum_assignment(cost)

    # 先全部置为 None（未匹配占位）
    pred_for_each_gt = [None] * G
    used_pred = set()
    matches = []

    for gi, pj in zip(row_ind, col_ind):
        iou = float(ious[gi, pj])
        if iou >= iou_threshold:
            pred_for_each_gt[gi] = pred_bboxes[pj]
            used_pred.add(pj)
            matches.append((gi, pj, iou))
        # 否则：这对匈牙利分到一起但 IoU 过低，视为未匹配，保留 None

    unmatched_gt = [gi for gi in range(G) if pred_for_each_gt[gi] is None]
    unmatched_pred = [pj for pj in range(P) if pj not in used_pred]

    return pred_for_each_gt, matches, unmatched_gt, unmatched_pred


def calc_box_iou(bb, BBGT):
    ovmax = -np.inf

    # compute overlaps
    # intersection
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
        (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
        + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
        - inters
    )

    overlaps = inters / uni
    ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)

    return ovmax


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calc_tpfp(gt_bboxes, bboxs):
    BBGT = np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4)

    blen = len(bboxs)
    if blen == 0:
        return [], []

    tp = np.zeros(blen)
    fp = np.zeros(blen)
    for i in range(blen):
        if bboxs[i] is None:
            fp[i] = 1
            continue
        bb = np.array(bboxs[i], np.float32)
        iou = calc_box_iou(bb, BBGT)
        # print("iou", iou)
        if iou >= 0.5:
            tp[i] = 1.0
        else:
            fp[i] = 1

    return (tp, fp)

    # recall = sum(tp) / (float(len(BBGT)) + 1e-6)
    # precision = sum(tp) / (float(blen) + 1e-6)
    # print(recall, precision)

    # fp = np.cumsum(fp)
    # tp = np.cumsum(tp)
    # rec = tp / (float(len(BBGT)) + 1e-6)

    # prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # ap = voc_ap(rec, prec, use_07_metric=False)

    # return ap


def cal_ap(tp_dict, fp_dict):
    ap_dict = {}
    for key in tp_dict.keys():
        tp = tp_dict[key]
        fp = fp_dict[key]
        len_of_gt = len(tp)
        # recall = sum(tp) / (float(len_of_gt) + 1e-6)
        # precision = sum(tp) / (float(len_of_gt) + 1e-6)
        # print(recall, precision)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / (float(len_of_gt) + 1e-6)

        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric=False)
        ap_dict[key] = ap

    return ap_dict


def tpfp(gt_bboxes, pred_img, gt_img, cls_name):
    """
    计算图像分割的IoU（Intersection over Union）

    参数:
        output: 模型输出的预测tensor，形状通常为(batch_size, channels, height, width)
        target: 目标标签tensor，形状与output相同
        valid_mask: 可选的有效区域掩码，形状与output的空间维度相同，bool类型

    返回:
        平均IoU分数
    """
    bboxes1, img_vis = ext_coor(pred_img, gt_img, cls_name)
    # bboxes2, output = extract_bbox(pred_img, gt_img)
    # bboxes = bboxes1 + bboxes2
    bboxes = bboxes1
    return calc_tpfp(gt_bboxes, bboxes)


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


# Adapted from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2)).cpu()
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask=None):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask=None):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask=None):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    print(f"min of {torch.min(output)}, max of {torch.max(output)}")
    print(f"min of {torch.min(target)}, max of {torch.max(target)}")
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    print(f"number of pixels {n}")
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss


# -------------------- Normals Metrics --------------------


def compute_cosine_error(pred_norm, gt_norm, masked=False):
    if len(pred_norm.shape) == 4:
        pred_norm = pred_norm.squeeze(0)
    if len(gt_norm.shape) == 4:
        gt_norm = gt_norm.squeeze(0)

    # shape must be [3,H,W]
    assert (gt_norm.shape[0] == 3) and (
        pred_norm.shape[0] == 3
    ), "Channel dim should be the first dimension!"
    # mask out the zero vectors, otherwise torch.cosine_similarity computes 90° as error
    if masked:
        ch, h, w = gt_norm.shape

        mask = torch.norm(gt_norm, dim=0) > 0

        pred_norm = pred_norm[:, mask.view(h, w)]
        gt_norm = gt_norm[:, mask.view(h, w)]

    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=0)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi  # (H, W)

    return (
        pred_error.view(-1).detach().cpu().numpy()
    )  # flatten so can directly input to compute_normal_metrics()


def mean_angular_error(cosine_error):
    return round(np.average(cosine_error), 4)


def median_angular_error(cosine_error):
    return round(np.median(cosine_error), 4)


def rmse_angular_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(np.sqrt(np.sum(cosine_error * cosine_error) / num_pixels), 4)


def sub5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 5) / num_pixels), 4)


def sub7_5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 7.5) / num_pixels), 4)


def sub11_25_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 11.25) / num_pixels), 4)


def sub22_5_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 22.5) / num_pixels), 4)


def sub30_error(cosine_error):
    num_pixels = cosine_error.shape[0]
    return round(100.0 * (np.sum(cosine_error < 30) / num_pixels), 4)


# -------------------- IID Metrics --------------------


def compute_iid_metric(pred, gt, target_name, metric_name, metric, valid_mask=None):
    # Shading and residual are up-to-scale. We first scale-align them to the gt
    # and map them to the range [0,1] for metric computation
    if target_name == "shading" or target_name == "residual":
        alignment_scale = compute_alignment_scale(pred, gt, valid_mask)
        pred = alignment_scale * pred
        # map to [0,1]
        pred, gt = quantile_map(pred, gt, valid_mask)

    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)
    if valid_mask is not None:
        if len(valid_mask.shape) == 3:
            valid_mask = valid_mask.unsqueeze(0)
        if metric_name == "psnr":
            return metric(pred[valid_mask], gt[valid_mask]).item()
        # for SSIM and LPIPs set the invalid pixels to zero
        else:
            invalid_mask = ~valid_mask
            pred[invalid_mask] = 0
            gt[invalid_mask] = 0

    return metric(pred, gt).item()


# compute least-squares alignment scale to align shading/residual prediction to gt
def compute_alignment_scale(pred, gt, valid_mask=None):
    pred = pred.squeeze()
    gt = gt.squeeze()
    assert pred.shape[0] == 3 and gt.shape[0] == 3, "First dim should be channel dim"

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
        pred = pred[valid_mask]
        gt = gt[valid_mask]

    A_flattened = pred.view(-1, 1)
    b_flattened = gt.view(-1, 1)
    # Solve the least squares problem
    x, residuals, rank, s = torch.linalg.lstsq(A_flattened.float(), b_flattened.float())
    return x


def quantile_map(pred, gt, valid_mask=None):
    pred = pred.squeeze()
    gt = gt.squeeze()
    assert gt.shape[0] == 3, "channel dim must be first dim"

    percentile = 90
    brightness_nth_percentile_desired = 0.8
    brightness = 0.3 * gt[0, :, :] + 0.59 * gt[1, :, :] + 0.11 * gt[2, :, :]

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
        brightness = brightness[valid_mask[0]]
    else:
        brightness = brightness.flatten()

    eps = 0.0001

    brightness_nth_percentile_current = torch.quantile(brightness, percentile / 100.0)

    if brightness_nth_percentile_current < eps:
        scale = 0
    else:
        scale = float(
            brightness_nth_percentile_desired / brightness_nth_percentile_current
        )

    # Apply scaling to ground truth and prediction
    gt_mapped = torch.clamp(scale * gt, 0, 1).unsqueeze(0)  # [1,3,H,W]
    pred_mapped = torch.clamp(scale * pred, 0, 1).unsqueeze(0)  # [1,3,H,W]

    return pred_mapped, gt_mapped


def rgb_absrel(img1: Image.Image, img2: Image.Image) -> float:
    """
    计算两张RGB图像的绝对相对误差(absrel)

    参数:
        img1: 第一张PIL RGB图像
        img2: 第二张PIL RGB图像

    返回:
        absrel值，越小表示相似度越高
    """
    # 确保图像尺寸相同
    if img1.size != img2.size:
        # 调整img2尺寸以匹配img1
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

    # 转换为numpy数组并归一化到[0, 1]范围
    arr1 = np.array(img1, dtype=np.float32) / 255.0
    arr2 = np.array(img2, dtype=np.float32) / 255.0

    # 计算绝对差
    abs_diff = np.abs(arr1 - arr2)

    # 处理接近零的值，避免除零错误
    eps = 1e-8
    arr2 = np.maximum(arr2, eps)

    # 计算相对误差并取平均值
    rel_error = abs_diff / arr2
    absrel = np.mean(rel_error)

    return absrel


def rgb_mse(img1: Image.Image, img2: Image.Image) -> float:
    """
    计算两张RGB图像的均方误差(MSE)

    参数:
        img1: 第一张PIL RGB图像
        img2: 第二张PIL RGB图像

    返回:
        MSE值，越小表示相似度越高
    """
    # 确保图像尺寸相同
    if img1.size != img2.size:
        # 调整img2尺寸以匹配img1
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

    # 转换为numpy数组
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    # 计算均方误差
    mse = np.mean((arr1 - arr2) ** 2)

    return mse


def rgb_delta1_acc(img1: Image.Image, img2: Image.Image) -> float:
    """
    计算两张RGB图像的delta1准确率

    参数:
        img1: 第一张PIL RGB图像
        img2: 第二张PIL RGB图像

    返回:
        delta1准确率(百分比)，越大表示相似度越高
    """
    # 确保图像尺寸相同
    if img1.size != img2.size:
        # 调整img2尺寸以匹配img1
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

    # 转换为numpy数组并归一化到[0, 1]范围
    arr1 = np.array(img1, dtype=np.float32) / 255.0
    arr2 = np.array(img2, dtype=np.float32) / 255.0

    # 处理接近零的值，避免除零错误
    eps = 1e-8
    arr1 = np.maximum(arr1, eps)
    arr2 = np.maximum(arr2, eps)

    # 计算比值
    ratio1 = arr1 / arr2
    ratio2 = arr2 / arr1

    # 取每个像素的最大值
    max_ratio = np.maximum(ratio1, ratio2)

    # 计算满足delta1条件的像素比例
    delta1_threshold = 1.25
    delta1_acc = np.mean(max_ratio < delta1_threshold) * 100  # 转换为百分比

    return delta1_acc


if __name__ == "__main__":
    import json
    import os
    from tqdm import tqdm

    steps = [2500, 5000]

    with open(
        "../../../../../data_sz/datasets/COCO2017/annotations/coco_val2017_800.jsonl",
        "r",
    ) as f:
        lines = f.readlines()

    for step in steps:
        tp_dict = {}
        fp_dict = {}
        for line in tqdm(lines):
            data = json.loads(line)
            idx = data["data_idx"]
            gt_bboxes = data["bbox"]
            cls_name = data["prompt"]

            if cls_name not in tp_dict:
                tp_dict[cls_name] = []
                fp_dict[cls_name] = []

            gt_img = Image.open(
                f"/root/paddlejob/workspace/output/zzf/codes/VisualSketchpad/ar_model/Bagel/results/20250901_170829_interleave_detection_infer_image/coco_val2017_800/step_000{step}/{idx}/detection_gt.png"
            )
            pred_img = Image.open(
                f"/root/paddlejob/workspace/output/zzf/codes/VisualSketchpad/ar_model/Bagel/results/20250901_170829_interleave_detection_infer_image/coco_val2017_800/step_000{step}/{idx}/detection_pred.png"
            )
            tp, fp = tpfp(gt_bboxes, pred_img, gt_img, cls_name)

            # 保证是可序列化的一维 int 列表
            tp = np.asarray(tp).reshape(-1).astype(int).tolist()
            fp = np.asarray(fp).reshape(-1).astype(int).tolist()

            tp_dict[cls_name].extend(tp)  # 累加 0/1
            fp_dict[cls_name].extend(fp)  # 注意用 extend 而不是 append

        ap_result = cal_ap(tp_dict, fp_dict)
        if isinstance(ap_result, tuple):
            ap_per_class, mean_ap = ap_result
        else:
            ap_per_class = ap_result
            mean_ap = (
                float(np.mean(list(ap_per_class.values())))
                if len(ap_per_class) > 0
                else 0.0
            )

        # 组织写入内容
        step_key = f"step_{step}"
        to_write = {
            "ap_per_class": {str(k): float(v) for k, v in ap_per_class.items()},
            "mAP": float(mean_ap),
        }
        if os.path.exists("./metrics.json"):
            with open("./metrics.json", "r", encoding="utf-8") as f:
                metrics_all = json.load(f)
        else:
            metrics_all = {}
        metrics_all[step_key] = to_write
        with open("./metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_all, f, ensure_ascii=False, indent=2)
