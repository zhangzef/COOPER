# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io, base64
import random
from PIL import Image, ImageFile, PngImagePlugin
import json
from .interleave_t2i_dataset import (
    InterleavedBaseIterableDataset,
    ParquetStandardIterableDataset,
)
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


INTRELEAVE_THINK_SYSTEM_PROMPT = """You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
<think> reasoning process here </think><answer> answer here </answer>"""


class InterleaveReasonIterableDataset(
    InterleavedBaseIterableDataset, ParquetStandardIterableDataset
):
    def parse_row(self, row):
        image_list = json.loads(row["image_list"])
        instruction_list = json.loads(
            row["instruction_list"]
        )  # len(image_list) == len(instruction_list) + 2
        start_idx = 0
        end_idx = len(image_list)

        data = self._init_data()
        data = self._add_text(
            data,
            INTRELEAVE_THINK_SYSTEM_PROMPT + "\n" + instruction_list[0],
            need_loss=False,
        )  # Question
        data = self._add_image(
            data,
            pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_list[0])))),
            need_loss=False,
            need_vae=True,
            need_vit=True,
        )  # Question Image
        start_idx += 1
        for i in range(start_idx, end_idx):
            data = self._add_text(data, instruction_list[i], need_loss=True)
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(base64.b64decode(image_list[i])))),
                need_loss=True,
                need_vae=False,
                need_vit=False,
            )
        data = self._add_text(
            data, instruction_list[-2] + "\n" + instruction_list[-1], need_loss=True
        )
        return data
