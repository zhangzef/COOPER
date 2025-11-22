# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Dict, Any, Tuple
import pyarrow.parquet as pq
import traceback
from ..distributed_iterable_dataset import DistributedIterableDataset
from ..parquet_utils import get_parquet_data_paths, init_arrow_pf_fs
from ..data_utils import pil_img2rgb
import json
import os
from PIL import Image, ImageFile, PngImagePlugin
import torch
import random


class InterleavedBaseIterableDataset(DistributedIterableDataset):
    def _init_data(self):
        data = {
            "sequence_plan": [],
            "text_ids_list": [],
            "image_tensor_list": [],
            "num_tokens": 0,
            "valid_mask": None,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        text_ids = self.tokenizer.encode(text)
        data["num_tokens"] += len(text_ids)
        data["text_ids_list"].append(text_ids)
        data["sequence_plan"].append(
            {
                "type": "text",
                "enable_cfg": int(enable_cfg),
                "loss": int(need_loss),
                "special_token_loss": 0,
                "special_token_label": None,
            }
        )
        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        if need_loss:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
            if self.dataset_name in ["ade20k"]:
                image_tensor = self.transform(
                    image, is_rgb=True, is_training=True, dataset=self.dataset_name
                )
            elif self.dataset_name in ["vkitti", "kitti", "hypersim", "nyu"]:
                image_tensor = self.transform(
                    image, is_rgb=False, is_training=True, dataset=self.dataset_name
                )
            elif self.dataset_name in ["detection"]:
                image_tensor = self.transform(
                    image, is_rgb=True, is_training=True, dataset=self.dataset_name
                )
            data["valid_mask"] = image_tensor.clone()[0, :, :]

            height, width = image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.transform.stride**2
            data["image_tensor_list"].append(image_tensor)

        if need_vae:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )

            image_tensor = self.transform(
                image, is_rgb=True, is_training=True, dataset=self.dataset_name
            )
            height, width = image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.transform.stride**2
            data["image_tensor_list"].append(image_tensor.clone())

        if need_vit:
            data["sequence_plan"].append(
                {
                    "type": "vit_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.vit_transform.stride**2
            data["image_tensor_list"].append(vit_image_tensor)
        return data


class JSONLStandardIterableDataset(DistributedIterableDataset):
    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        vit_transform,
        data_dir_list,
        jsonl_path_list,
        num_used_data,
        local_rank: int = 0,
        world_size: int = 1,
        num_workers: int = 8,
        data_status: Optional[List[List[int]]] = None,
        shuffle_lines: bool = False,
        shuffle_seed: int = 42,
    ):
        """
        Args:
            dataset_name: 数据集名称，用于标识
            jsonl_path_list: JSONL 文件路径列表
            num_used_data: 每个 JSONL 文件中使用的数据行数
            local_rank: 当前进程的 local_rank
            world_size: 总进程数（world_size）
            num_workers: DataLoader 的 worker 数量
            data_status: 用于恢复训练时记录当前读取进度
            shuffle_lines: 是否对 JSONL 文件中的数据行进行打乱
            shuffle_seed: 打乱数据行时使用的随机种子
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        print(f"[{self.dataset_name}] init dataset")

        self.jsonl_path_list = jsonl_path_list
        self.num_used_data = num_used_data
        self.data_status = data_status
        self.shuffle_lines = shuffle_lines
        self.shuffle_seed = shuffle_seed
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.transform = transform

        # 初始化数据路径
        self.data_paths = self.get_data_paths(
            jsonl_path_list=jsonl_path_list,
            data_dir_list=data_dir_list,
            num_used_data=num_used_data,
            shuffle_lines=shuffle_lines,
            shuffle_seed=shuffle_seed,
        )

        # 设置 epoch，用于打乱整个数据路径
        self.set_epoch()

    def get_data_paths(
        self,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        shuffle_lines,
        shuffle_seed,
    ):
        """
        构建数据路径：(file_path, line_index)
        """
        data_paths = []

        for jsonl_path, image_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                raw_data = f.readlines()

            if shuffle_lines:
                rng = random.Random(shuffle_seed)
                rng.shuffle(raw_data)

            raw_data = raw_data[:num_data_point]

            for line_idx, line in enumerate(raw_data):
                data_paths.append((line_idx, image_dir, line))

        return data_paths

    def parse_row(self, image_dir, row):
        """
        子类需实现此方法，用于解析每行 JSON 数据
        """
        raise NotImplementedError

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            start_idx = self.data_status[worker_id] + 1
        else:
            start_idx = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at index#{start_idx}"
        )

        while True:
            data_paths_subset = data_paths_per_worker[start_idx:]

            for idx, (line_idx, image_dir, line) in enumerate(
                data_paths_subset, start=start_idx
            ):
                try:
                    data = json.loads(line.strip())
                    parsed_data = self.parse_row(image_dir, data)
                    if not parsed_data:
                        continue

                    parsed_data["data_indexes"] = {
                        "data_indexes": [idx, line_idx],
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                except Exception as e:
                    print(f"Error parsing line {line_idx}: {e}")
                    continue

                yield parsed_data

            start_idx = 0
            print(
                f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}"
            )
