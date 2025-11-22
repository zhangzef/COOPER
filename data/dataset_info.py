# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import InterleaveReasonIterableDataset
from .depth_datasets import InterleaveDepthIterableDataset


DATASET_REGISTRY = {
    "reason_interleave_dataset": InterleaveReasonIterableDataset,
    "vkitti": InterleaveDepthIterableDataset,
    "hypersim": InterleaveDepthIterableDataset,
    "ade20k": InterleaveDepthIterableDataset,
}


DATASET_INFO = {
    "reason_interleave_dataset": {
        "reason_interleave_dataset": {
            "data_dir": "-",
            "jsonl_path": "./datasets/COOPER_reasoning_train_set/Interleaved_SFT.jsonl",
            "num_total_samples": 6901,
        }
    },
    "vkitti": {
        "vkitti": {
            "data_dir": "./datasets/vkitti/",
            "jsonl_path": "./datasets/vkitti/train.jsonl",
            "num_total_samples": 20148,
        }
    },
    "hypersim": {
        "hypersim": {
            "data_dir": "./datasets/Hypersim/",
            "jsonl_path": "./datasets/Hypersim/filename_list_train_filtered.jsonl",
            "num_total_samples": 53885,
        }
    },
    "ade20k": {
        "ade20k": {
            "data_dir": "./datasets/ADE20K_SEG",
            "jsonl_path": "./datasets/ADE20K_SEG/ade20k_train_identify.jsonl",
            "num_total_samples": 25574,
        }
    },
}
