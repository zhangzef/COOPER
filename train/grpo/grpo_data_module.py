# -*- coding: utf-8 -*-

import json
from torch.utils.data import Dataset
from PIL import Image
import random


class GRPODataset(Dataset):
    def __init__(self, jsonl_path, image_root):
        super().__init__()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data_list = f.readlines()
        random.seed(42)
        random.shuffle(self.data_list)
        self.image_root = image_root

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        data = json.loads(self.data_list[index])
        question = data["question"]
        answer = data["answer"]
        img = Image.open(f"{self.image_root}/{data['image']}").convert("RGB")
        flag = data["flag"] if "flag" in data else "boundary"
        return {
            "question": question,
            "solution": answer,
            "image": img,
            "data_id": data["data_id"],
            "flag": flag,
        }
