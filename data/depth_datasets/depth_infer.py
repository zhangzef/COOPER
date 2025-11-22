import json
import os
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset
import random
from ..transforms import DepthImageTransform
from pycocotools import mask as coco_mask
import torch


DEPTH_PROMPT = "<depth-estimation>Estimate the depth of the image and generate the depth map.</depth-estimation>"
SEGMENTATION_PROMPT = "<segmentation>Segment the objects in the image with different colors.</segmentation>"
DETECTION_PROMPT = "<detection>Detect and box all the {object} in the image with a red box.</detection>"
PROMPT = ""


class DepthInferDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir,
        jsonl_path,
        dataset_name,
        shuffle_lines: bool = False,
        shuffle_seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.jsonl_path = jsonl_path
        self.shuffle_lines = shuffle_lines
        self.shuffle_seed = shuffle_seed
        self.transform = transform
        self.image_dir = image_dir

        # 初始化数据路径
        self.data_paths = self.get_data_paths(
            jsonl_path=jsonl_path,
            shuffle_lines=shuffle_lines,
            shuffle_seed=shuffle_seed,
        )

    def get_data_paths(
        self,
        jsonl_path,
        shuffle_lines,
        shuffle_seed,
    ):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_data = f.readlines()
        if shuffle_lines:
            rng = random.Random(shuffle_seed)
            rng.shuffle(raw_data)
        return raw_data

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        line = self.data_paths[index]
        data = json.loads(line.strip())
        rgb_image = Image.open(os.path.join(self.image_dir, data["rgb_image"]))
        gen_image_tensor = rgb_image
        if self.dataset_name in ["vkitti", "kitti", "nyu", "hypersim"]:
            PROMPT = DEPTH_PROMPT
            depth_image = Image.open(os.path.join(self.image_dir, data["depth_image"]))
            if self.dataset_name in ["vkitti", "kitti"]:
                rgb_image = DepthImageTransform.kitti_benchmark_crop(rgb_image)
                depth_image = DepthImageTransform.kitti_benchmark_crop(depth_image)
            (depth_image_tensor, valid_mask) = self.transform(
                depth_image, is_rgb=False, is_training=False, dataset=self.dataset_name
            )
            gen_image_tensor = depth_image_tensor

        return {
            "text": PROMPT,
            "data_id": data["data_id"],
            "rgb_image": rgb_image,
            "gen_image": gen_image_tensor,
            "valid_mask": valid_mask,
        }


class SegmentationInferDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir,
        jsonl_path,
        dataset_name,
        shuffle_lines: bool = False,
        shuffle_seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.jsonl_path = jsonl_path
        self.shuffle_lines = shuffle_lines
        self.shuffle_seed = shuffle_seed
        self.transform = transform
        self.image_dir = image_dir

        # 初始化数据路径
        self.data_paths = self.get_data_paths(
            jsonl_path=jsonl_path,
            shuffle_lines=shuffle_lines,
            shuffle_seed=shuffle_seed,
        )

    def get_data_paths(
        self,
        jsonl_path,
        shuffle_lines,
        shuffle_seed,
    ):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_data = f.readlines()
        if shuffle_lines:
            rng = random.Random(shuffle_seed)
            rng.shuffle(raw_data)
        return raw_data

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        line = self.data_paths[index]
        data = json.loads(line.strip())
        rgb_image = Image.open(os.path.join(self.image_dir, data["rgb_image"]))
        detection_image = Image.open(os.path.join(self.image_dir, data["seg_image"]))
        PROMPT = SEGMENTATION_PROMPT

        return {
            "text": PROMPT,
            "data_id": data["data_idx"],
            "rgb_image": rgb_image,
            "gen_image": detection_image,
            "image_id": data["rgb_image"].rsplit("/", 1)[-1].replace(".jpg", ""),
        }


class DetectionInferDataset(Dataset):
    def __init__(
        self,
        transform,
        image_dir,
        jsonl_path,
        dataset_name,
        shuffle_lines: bool = False,
        shuffle_seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.jsonl_path = jsonl_path
        self.shuffle_lines = shuffle_lines
        self.shuffle_seed = shuffle_seed
        self.transform = transform
        self.image_dir = image_dir

        # 初始化数据路径
        self.data_paths = self.get_data_paths(
            jsonl_path=jsonl_path,
            shuffle_lines=shuffle_lines,
            shuffle_seed=shuffle_seed,
        )[:80]

    def get_data_paths(
        self,
        jsonl_path,
        shuffle_lines,
        shuffle_seed,
    ):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw_data = f.readlines()
        if shuffle_lines:
            rng = random.Random(shuffle_seed)
            rng.shuffle(raw_data)
        return raw_data

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        line = self.data_paths[index]
        data = json.loads(line.strip())
        rgb_image = Image.open(os.path.join(self.image_dir, data["rgb_image"]))
        detection_image = Image.open(os.path.join(self.image_dir, data["det_image"]))
        PROMPT = DETECTION_PROMPT.format(object=data["prompt"])

        return {
            "text": PROMPT,
            "data_id": data["data_idx"],
            "rgb_image": rgb_image,
            "gen_image": detection_image,
            "gt_bboxes": data["bbox"],
            "cls_name": data["prompt"],
            "image_id": data["rgb_image"].rsplit("/", 1)[-1].replace(".jpg", ""),
        }


def collate_fn_depth(batch):
    text = [sample["text"] for sample in batch]
    data_id = [sample["data_id"] for sample in batch]
    rgb_image = [sample["rgb_image"] for sample in batch]
    gen_image = [sample["gen_image"] for sample in batch]
    valid_mask = [sample["valid_mask"] for sample in batch]

    return {
        "text": text,
        "data_id": data_id,
        "rgb_image": rgb_image,
        "gen_image": gen_image,
        "valid_mask": valid_mask,
    }


def collate_fn_detection(batch):
    text = [sample["text"] for sample in batch]
    data_id = [sample["data_id"] for sample in batch]
    rgb_image = [sample["rgb_image"] for sample in batch]
    gen_image = [sample["gen_image"] for sample in batch]
    gt_bboxes = [sample["gt_bboxes"] for sample in batch]
    cls_name = [sample["cls_name"] for sample in batch]
    image_id = [sample["image_id"] for sample in batch]

    return {
        "text": text,
        "data_id": data_id,
        "rgb_image": rgb_image,
        "gen_image": gen_image,
        "gt_bboxes": gt_bboxes,
        "cls_name": cls_name,
        "image_id": image_id,
    }


def collate_fn_segmentation(batch):
    text = [sample["text"] for sample in batch]
    data_id = [sample["data_id"] for sample in batch]
    rgb_image = [sample["rgb_image"] for sample in batch]
    gen_image = [sample["gen_image"] for sample in batch]
    image_id = [sample["image_id"] for sample in batch]

    return {
        "text": text,
        "data_id": data_id,
        "rgb_image": rgb_image,
        "gen_image": gen_image,
        "image_id": image_id,
    }
