# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io, base64
import random
from PIL import Image, ImageFile, PngImagePlugin
import json
from .interleave_base import (
    InterleavedBaseIterableDataset,
    JSONLStandardIterableDataset,
)
from ..data_utils import pil_img2rgb
import os
import random
import torchvision.transforms as transforms
from ..transforms import DepthImageTransform
from pycocotools import mask as coco_mask
import numpy as np
import random


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

DEPTH_PROMPT = "<depth-estimation>Estimate the depth of the image and generate the depth map.</depth-estimation>"
SEGMENTATION_PROMPT = "<segmentation>Segment the objects in the image with different colors.</segmentation>"
DETECTION_PROMPT = "<detection>Detect and box all the {object} in the image with a red box.</detection>"
PROMPT = ""


def array_statistics(arr):
    """
    计算并打印NumPy数组的统计值：众数、均值、方差以及不同元素的个数字典

    参数:
        arr (np.ndarray): 输入的NumPy数组（支持多维数组）
    """
    # 将数组展平为1维，方便统计所有元素
    flat_arr = arr.flatten()

    # 获取唯一元素及其出现次数
    unique_elements, counts = np.unique(flat_arr, return_counts=True)

    # 计算不同元素的个数字典
    element_count_dict = {elem: count for elem, count in zip(unique_elements, counts)}

    # 计算众数（出现次数最多的元素）
    # 处理可能存在多个众数的情况
    max_count = np.max(counts)
    modes = unique_elements[counts == max_count]

    # 打印统计结果
    print("数组统计信息：")
    print(f"shape {arr.shape}")
    print(f"不同元素的个数: {len(unique_elements)}")
    print("不同元素的个数字典:")
    for elem, count in element_count_dict.items():
        print(f"  {elem}: {count}次")
    print(f"众数: {modes} (出现次数: {max_count})")


class InterleaveDepthIterableDataset(
    InterleavedBaseIterableDataset, JSONLStandardIterableDataset
):
    def parse_row(self, image_dir, row):
        rgb_image = Image.open(os.path.join(image_dir, row["rgb_image"])).convert("RGB")
        gen_image = None

        if self.dataset_name in ["vkitti", "kitti", "hypersim", "nyu"]:
            depth_image = Image.open(os.path.join(image_dir, row["depth_image"]))
            PROMPT = DEPTH_PROMPT
            if self.dataset_name in ["vkitti", "kitti"]:
                rgb_image = DepthImageTransform.kitti_benchmark_crop(rgb_image)
                depth_image = DepthImageTransform.kitti_benchmark_crop(depth_image)
            gen_image = depth_image
            if random.random() < 0.5:  # 50%概率翻转
                rgb_image = transforms.functional.hflip(rgb_image)
                gen_image = transforms.functional.hflip(gen_image)
        elif self.dataset_name in ["ade20k"]:
            PROMPT = SEGMENTATION_PROMPT
            seg_image = Image.open(os.path.join(image_dir, row["seg_image"]))
            gen_image = seg_image
            if random.random() < 0.5:  # 50%概率翻转
                rgb_image = transforms.functional.hflip(rgb_image)
                gen_image = transforms.functional.hflip(gen_image)
        elif self.dataset_name in ["detection"]:
            PROMPT = DETECTION_PROMPT.format(object=row["prompt"])
            gen_image = depth_image = Image.open(
                os.path.join(image_dir, row["det_image"])
            )

        data = self._init_data()
        data = self._add_image(
            data,
            rgb_image,
            need_loss=False,
            need_vae=True,
            need_vit=True,
        )  # Question Image
        data = self._add_text(
            data,
            PROMPT,
            need_loss=False,
        )
        data = self._add_image(
            data,
            gen_image,
            need_loss=True,
            need_vae=False,
            need_vit=False,
        )
        return data
