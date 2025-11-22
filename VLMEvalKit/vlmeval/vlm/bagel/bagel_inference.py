from __future__ import annotations

import os
import io
import base64
import requests
from PIL import Image
from accelerate import (
    load_checkpoint_and_dispatch,
    init_empty_weights,
)
import torch
from ..base import BaseModel
from .data.transforms import ImageTransform
from .data.data_utils import pil_img2rgb, add_special_tokens
from .modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from .modeling.qwen2 import Qwen2Tokenizer
from .modeling.autoencoder import load_ae
from .inferencer import InterleaveInferencer
from datetime import datetime
import random
import string
import json


def ensure_image_rgb(image: str) -> Image.Image:
    """
    检查图像输入是否合法，并返回 RGB 格式的 PIL 图像对象。
    支持：
      - http(s):// URLs
      - file:// 路径
      - 本地路径
      - data:image;base64, 图像数据
    """
    if image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return img

    if image.startswith("file://"):
        path = image[7:]
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
        return Image.open(path).convert("RGB")

    if image.startswith("data:image"):
        header, base64_data = image.split(",", 1)
        img_bytes = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if os.path.exists(image):
        return Image.open(image).convert("RGB")

    raise ValueError(f"Invalid image path or URL: {image}")


def process_lists(input_list, output_list, target_dir):
    """
    将输入列表和输出列表合并为字典，处理其中的图像并保存为JSON文件

    参数:
        input_list: 包含字符串和PIL Image对象的列表
        output_list: 包含字符串和PIL Image对象的列表
        target_dir: 目标目录，用于创建新文件夹和保存文件
    """

    # 合并为字典
    data_dict = {"input_list": input_list, "output_list": output_list}

    # 生成不重复的文件夹名称 (时间戳 + 随机字符串)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_str = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    folder_name = f"processed_data_{timestamp}_{random_str}"
    folder_path = os.path.join(target_dir, folder_name)

    # 创建文件夹
    os.makedirs(folder_path, exist_ok=True)

    # 处理字典中的图像，保存并替换为路径
    image_counter = 1

    for key in data_dict.keys():
        for i in range(len(data_dict[key])):
            if isinstance(data_dict[key][i], Image.Image):
                # 保存图像
                img_filename = f"{key}_image_{image_counter}.png"
                img_path = os.path.join(folder_path, img_filename)
                data_dict[key][i].save(img_path)
                data_dict[key][i] = img_path
                image_counter += 1

    # 保存字典为JSON文件
    json_filename = "data_dict.json"
    json_path = os.path.join(folder_path, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

    return folder_path, json_path


class BagelInference(BaseModel):
    def __init__(
        self,
        model_config_path: str,
        model_param_path: str,
        reasoning_mode: str,  # text, image, interleave
        is_thinking: bool = True,
        max_inter_num=3,
        max_think_token_n: int = 2048,
        max_new_tokns: int = 1024,
        is_ema: bool = False,  # the model save type, if sft, then use ema model, if rl, then use not ema model
        visual_gen: bool = True,
        visual_und: bool = True,
        do_sample=False,
        text_temperature=1.0,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
        repetition_penalty: float = 1.0,
        verbose=False,
        save_file=None,  # if None, then no save the reasoning process
    ):
        super().__init__()
        self.verbose = verbose
        self.text_temperature = text_temperature
        self.do_sample = do_sample
        self.max_new_tokns = max_new_tokns
        self.reasoning_mode = reasoning_mode
        assert self.reasoning_mode in ["text", "image", "interleave"]
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = f"cuda:{local_rank}"
        self.save_file = save_file

        llm_config = Qwen2Config.from_json_file(
            os.path.join(model_config_path, "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(model_config_path, "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        self.vae_model, vae_config = load_ae(
            local_path=os.path.join(model_config_path, "ae.safetensors")
        )
        self.vae_model = self.vae_model.to(
            device=self.device,  # 与主模型同卡
            dtype=torch.bfloat16,  # ↙️ 关键：统一成 bfloat16
        ).eval()
        config = BagelConfig(
            visual_gen=visual_gen,
            visual_und=visual_und,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True
            )
        tokenizer = Qwen2Tokenizer.from_pretrained(model_config_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(tokenizer)
        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(518, 224, 14)
        self.model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(
                model_param_path, "ema.safetensors" if is_ema else "model.safetensors"
            ),
            device_map={"": self.device},
            offload_buffers=False,
            dtype=torch.bfloat16,
        )
        self.model = self.model.eval()
        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids,
            device=self.device,
        )

        self.inference_hyper = dict(
            max_think_token_n=max_think_token_n,
            max_new_tokns=self.max_new_tokns,
            do_sample=self.do_sample,
            text_temperature=self.text_temperature,
            max_inter_num=max_inter_num,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            timestep_shift=timestep_shift,
            num_timesteps=num_timesteps,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            repetition_penalty=repetition_penalty,
            is_thinking=is_thinking,
        )
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):

        input_lists = []
        for s in message:
            if s["type"] == "image":
                item = ensure_image_rgb(s["value"])
            elif s["type"] == "text":
                item = s["value"]
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            input_lists.append(item)

        if self.reasoning_mode == "interleave":
            output_list = self.inferencer.interleave_reason_tool_condition(
                input_lists=input_lists,
                **self.inference_hyper,
            )
            response = output_list[-1].strip()
        elif self.reasoning_mode == "image":
            output_list = self.inferencer.image_reason_tool_condition(
                input_lists=input_lists, **self.inference_hyper
            )
            response = output_list[-1].strip()
        else:
            output_list = self.inferencer.text_reason(
                input_lists=input_lists, **self.inference_hyper
            )
            response = output_list[-1].split("</think>")[-1].strip()

        if self.verbose:
            print(f"\033[32m{output_list}\033[0m")

        if self.save_file is not None:
            process_lists(
                input_list=input_lists,
                output_list=output_list,
                target_dir=os.path.join("./stored_rl", self.save_file),
            )
        return response
