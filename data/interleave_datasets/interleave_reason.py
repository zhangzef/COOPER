# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io, base64
import random
from PIL import Image, ImageFile, PngImagePlugin
import json
from .interleave_t2i_dataset import (
    InterleavedBaseIterableDataset,
    ParquetStandardIterableDataset,
    JSONLStandardIterableDataset,
)
from ..data_utils import pil_img2rgb
import os
from io import BytesIO


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


INTERLEAVE_REASON_SYSTEM_PROMPT = """You are a specialized multimodal assistant. Your purpose is to solve visual question answering tasks by thinking step-by-step with your skill of depth estimation and semantic segmentation.

# Skills

You can prompt yourself to generate depth-estimation/segmentation by <depth-estimation>...</depth-estimation> / <segmentation>...</segmentation> as follows:

<depth-estimation>Estimate the depth of the image and generate the depth map.</depth-estimation>
OR
<segmentation>Segment the objects in the image with different colors.</segmentation>

# Instruction

1. In each turn, you should start with <think> tag. In this tag, you need to conduct a step-by-step reasoning process about the image and question and evaluate whether tool use would be helpful and give the reason. If received generated results, you also need to analyze them.
2. Please note that the result of the generated depth-estimation/segmentation are not always accurate. Please carefully check whether they are helpful for answering questions or whether they are accurate.
3. If you think depth-estimation/segmentation is useful, call to generate them by <depth-estimation>...</depth-estimation> / <segmentation>...</segmentation>. 
4. If you think no more to generate, you can answer in <answer> tag. You need to provide a concise summary of your reasoning process that leads to the final answer. Besides, you also need to put a simple and direct answer in \\boxed{{}} for verification.
5. Try to use the tools as much as possible.

The structure of your response should be like this:
<think> ... </think>
<depth-estimation> ... </depth-estimation> / <segmentation> ... </segmentation>

OR

<think> ... </think>
<answer> ... </answer>
"""

TEXT_REASON_SYSTEM_PROMPT = """You are a specialized multimodal language model. Your purpose is to solve visual question answering tasks by thinking step-by-step.

# Instruction

1. You should first think with <think> tag. In this tag, you need to conduct a step-by-step reasoning process about the image and question.
2. If you think that you could find the right answer, you can answer in <answer> tag. You need to provide a concise summary of your reasoning process that leads to the final answer. Besides, you also need to put a simple and direct answer in \\boxed{{}} for verification.

The structure of your response should be like this:
<think> thinking process satisfying the Instruction 1 </think>
<answer> answer satisfying the Instruction 2 </answer>
"""


def base64_to_image(base64_str):
    """
    将base64字符串转换为PIL Image并保存到本地

    参数:
        base64_str (str): 图像的base64编码字符串
    """
    # 移除base64字符串可能包含的前缀（如'data:image/jpeg;base64,'）
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]

    # 解码base64字符串为字节数据
    image_bytes = base64.b64decode(base64_str)

    # 将字节数据转换为PIL Image对象
    return Image.open(BytesIO(image_bytes)).convert("RGB")


class InterleaveReasonIterableDataset(
    InterleavedBaseIterableDataset, JSONLStandardIterableDataset
):
    def parse_row(self, image_dir, row):
        interleave_reason_list = row["interleaved_reason"]

        data = self._init_data()
        data = self._add_text(
            data,
            INTERLEAVE_REASON_SYSTEM_PROMPT,
            need_loss=False,
        )  # Question

        for idx in range(len(interleave_reason_list)):
            if interleave_reason_list[idx]["type"] == "text":
                data = self._add_text(
                    data,
                    interleave_reason_list[idx]["value"],
                    need_loss=(
                        True
                        if interleave_reason_list[idx]["need_loss"] == "true"
                        else False
                    ),
                )
            else:
                data = self._add_image(
                    data,
                    base64_to_image(interleave_reason_list[idx]["value"]),
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                )
        return data


class TextReasonIterableDataset(
    InterleavedBaseIterableDataset, JSONLStandardIterableDataset
):
    def parse_row(self, image_dir, row):
        interleave_reason_list = row["interleaved_reason"]

        data = self._init_data()
        data = self._add_text(
            data,
            TEXT_REASON_SYSTEM_PROMPT,
            need_loss=False,
        )  # Question

        for idx in range(len(interleave_reason_list)):
            if interleave_reason_list[idx]["type"] == "text":
                data = self._add_text(
                    data,
                    interleave_reason_list[idx]["value"],
                    need_loss=(
                        True
                        if interleave_reason_list[idx]["need_loss"] == "true"
                        else False
                    ),
                )
            else:
                data = self._add_image(
                    data,
                    base64_to_image(interleave_reason_list[idx]["value"]),
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                )
        return data
