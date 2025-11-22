# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, field
from time import time
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from inferencer import InterleaveInferencer
from data.depth_datasets import (
    DepthInferDataset,
    metric,
    DetectionInferDataset,
    SegmentationInferDataset,
)
from data.depth_datasets.depth_infer import (
    collate_fn_depth,
    collate_fn_detection,
    collate_fn_segmentation,
)
from data.depth_datasets.metric import tpfp, cal_ap, rgb_absrel, rgb_delta1_acc, rgb_mse
from data.depth_datasets.metric import MetricTracker
from data.depth_datasets.alignment import align_depth_least_square
from data.transforms import (
    DepthImageTransform,
    ImageTransform,
    colorize_depth_maps,
    chw2hwc,
)
from torch.utils.data.distributed import DistributedSampler
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, dispatch_model
from typing import List
import os, json
import safetensors.torch as st


@dataclass
class EvalArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."},
    )
    model_path_param: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."},
    )
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."},
    )
    prefetch_factor: int = field(
        default=2,
        metadata={
            "help": "How many batches each DataLoader worker pre-loads in advance."
        },
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."},
    )
    data_seed: int = field(
        default=42,
        metadata={
            "help": "Seed used when shuffling / sampling data shards to ensure reproducibility."
        },
    )
    valid_data_path: str = field(
        default=None, metadata={"help": "Path to validation split."}
    )
    valid_image_path: str = field(
        default=None,
        metadata={"help": "Path to validation images."},
    )
    valid_metrics: List[str] = field(
        default_factory=lambda: [  # 用lambda生成默认列表（避免可变类型共享）
            "abs_relative_difference",
            "squared_relative_difference",
            "rmse_linear",
            "rmse_log",
            "log10",
            "delta1_acc",
            "delta2_acc",
            "delta3_acc",
            "i_rmse",
            "silog_rmse",
        ],
        metadata={
            "help": "List of validation metrics (space-separated, e.g., iou rmse_linear)"
        },
    )
    valid_alignment: str = field(
        default="least_square",
        metadata={"help": "Alignment method for validation depth maps."},
    )
    save_to_dir: str = field(
        default=None,
        metadata={"help": "Directory where to save generated images."},
    )
    num_timesteps: int = field(
        default=50,
        metadata={"help": "Number of timesteps for depth prediction."},
    )
    valid_dataset_name: str = field(
        default="vkitti",
        metadata={"help": "Name of the validation dataset."},
    )


def draw_transparent_mask(image, mask, color=(255, 0, 0), alpha=128):
    """
    在PIL图像上根据mask矩阵绘制半透明的mask

    参数:
        image: PIL Image对象，原始图像
        mask: 布尔值numpy数组，形状为(H, W)，与图像尺寸匹配
              True表示需要绘制mask的位置
        color: 元组，mask的RGB颜色，默认为红色(255, 0, 0)
        alpha: 整数，透明度，0-255之间，默认为128（半透明）

    返回:
        PIL Image对象，带有半透明mask的图像
    """
    # 确保mask是布尔值数组
    mask = np.asarray(mask).astype(bool)

    # 检查mask与图像尺寸是否匹配
    if mask.shape != (image.height, image.width):
        raise ValueError(
            f"mask尺寸({mask.shape})与图像尺寸({image.height}, {image.width})不匹配"
        )

    # 将图像转换为RGBA以便处理透明度
    img_rgba = image.convert("RGBA")

    # 将图像转换为numpy数组以便操作
    img_array = np.array(img_rgba)

    # 创建mask的RGBA数组
    mask_rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    mask_rgba[..., 0] = color[0]  # R通道
    mask_rgba[..., 1] = color[1]  # G通道
    mask_rgba[..., 2] = color[2]  # B通道
    mask_rgba[..., 3] = alpha  # Alpha通道

    # 只在mask为True的位置应用半透明覆盖
    # 使用alpha混合公式: 结果 = 原图 * (1 - alpha/255) + mask * (alpha/255)
    alpha_normalized = alpha / 255.0
    img_array[mask] = (
        img_array[mask] * (1 - alpha_normalized) + mask_rgba[mask] * alpha_normalized
    ).astype(np.uint8)

    # 将numpy数组转换回PIL图像
    result = Image.fromarray(img_array)

    # 如果原图不是RGBA模式，转换回原图的模式
    if image.mode != "RGBA":
        result = result.convert(image.mode)

    return result


@torch.no_grad()
def validate(
    inferencer,
    valid_loader,
    metric_funcs,
    alignment,
    metric_tracker,
    curr_step,
    dataset_name,
    transform,
    num_timesteps=50,
    save_to_dir=None,
):
    with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        for batch_idx, batch in enumerate(tqdm(valid_loader)):
            # 数据加载和预处理
            rgb_image = batch["rgb_image"][0]
            gen_tensor = batch["gen_image"][0]
            data_id = batch["data_id"][0]
            text = batch["text"][0]
            valid_mask = batch["valid_mask"][0]

            # 深度预测推理
            gen_tensor_pred = inferencer.depth_map_generation(
                input_lists=[rgb_image, text],
                image_shapes=rgb_image.size,
                num_timesteps=num_timesteps,
            )  # range (0 ~ 1) shape(H, W)

            if dataset_name in ["vkitti", "kitti", "nyu", "hypersim"]:
                min_depth, max_depth = DepthImageTransform.get_depth_range(dataset_name)
                gen_tensor_to_save = gen_tensor_pred.clone().cpu().float()

                # 深度对齐
                if "least_square" == alignment:
                    gen_tensor_pred, scale, shift = align_depth_least_square(
                        gt_arr=gen_tensor.float(),
                        pred_arr=gen_tensor_pred.cpu().float(),
                        valid_mask_arr=valid_mask,
                        return_scale_shift=True,
                        max_resolution=None,
                    )
                else:
                    raise RuntimeError(f"Unknown alignment type: {alignment}")

                # 裁剪到数据集的最大最小值
                gen_pred = gen_tensor_pred.clamp(min=min_depth, max=max_depth)
                gen_pred = gen_pred.clamp(min=1e-6, max=None)
            elif dataset_name in ["segmentation"]:
                gen_pred = (gen_tensor_pred >= 0.5).cpu()

            # 评估指标计算
            sample_metric = []
            for met_func in metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(
                    gen_pred,
                    gen_tensor,
                    valid_mask=valid_mask,
                ).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)

            # 保存结果
            if save_to_dir is not None:
                curr_dir = os.path.join(
                    save_to_dir, f"step_{curr_step}", dataset_name, data_id
                )
                os.makedirs(curr_dir, exist_ok=True)
                rgb_image.save(os.path.join(curr_dir, "rgb.png"))
                if dataset_name in ["vkitti", "kitti", "nyu", "hypersim"]:
                    depth_colored = (
                        colorize_depth_maps(gen_tensor_to_save, 0, 1).squeeze().numpy()
                    )  # [3, H, W], value in (0, 1)
                    depth_colored = (depth_colored * 255).astype(np.uint8)
                    depth_colored_hwc = chw2hwc(depth_colored)
                    depth_colored_img = Image.fromarray(depth_colored_hwc)
                    depth_colored = (
                        colorize_depth_maps(gen_tensor, 0, 1).squeeze().numpy()
                    )  # [3, H, W], value in (0, 1)
                    depth_colored = (depth_colored * 255).astype(np.uint8)
                    depth_colored_hwc = chw2hwc(depth_colored)
                    depth_gt_img = Image.fromarray(depth_colored_hwc)
                    depth_colored_img.save(os.path.join(curr_dir, "depth_pred.png"))
                    depth_gt_img.save(os.path.join(curr_dir, "depth_gt.png"))
                elif dataset_name in ["segmentation"]:
                    rgb_image = transform.resize_transform(rgb_image, img_num=1)
                    pred_image = draw_transparent_mask(rgb_image, gen_pred)
                    label_image = draw_transparent_mask(rgb_image, gen_tensor)
                    pred_image.save(os.path.join(curr_dir, "seg_pred.png"))
                    label_image.save(os.path.join(curr_dir, "seg_gt.png"))

        global_metrics = metric_tracker.all_gather()
        if dist.get_rank() == 0:
            print("All results:")
            for key, val in global_metrics.items():
                print(f"{key}: {val:.4f}")
            if save_to_dir is not None:
                metrics_path = os.path.join(save_to_dir, "metrics.json")
                if not os.path.exists(metrics_path):
                    with open(metrics_path, "w") as f:
                        json.dump(
                            {f"step_{curr_step}": {dataset_name: global_metrics}}, f
                        )
                else:
                    with open(metrics_path, "r") as f:
                        metrics_dict = json.load(f)

                    if not f"step_{curr_step}" in metrics_dict.keys():
                        metrics_dict[f"step_{curr_step}"] = {
                            dataset_name: global_metrics
                        }
                    else:
                        metrics_dict[f"step_{curr_step}"] = {
                            dataset_name: global_metrics
                        }
                    with open(metrics_path, "w") as f:
                        json.dump(metrics_dict, f)
            return global_metrics
        else:
            return None


@torch.no_grad()
def validate_detection(
    inferencer,
    valid_loader,
    curr_step,
    num_timesteps=50,
    save_to_dir=None,
    dataset_name="ade20k",
):
    tp_dict = {}
    fp_dict = {}
    rgb_absrel_list = []
    rgb_delta1_acc_list = []
    rgb_mse_list = []

    with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        for batch_idx, batch in enumerate(tqdm(valid_loader)):
            # ===== 数据加载 =====
            rgb_image = batch["rgb_image"][0]
            gen_image = batch["gen_image"][0]
            data_id = batch["data_id"][0]
            text = batch["text"][0]
            if dataset_name == "detection":
                gt_bboxes = batch["gt_bboxes"][0]
                cls_name = batch["cls_name"][0]
                if cls_name not in tp_dict:
                    tp_dict[cls_name] = []
                    fp_dict[cls_name] = []
            image_id = batch["image_id"][0]

            # ===== 推理 =====
            pred_img = inferencer.image_generation_edit(
                input_list=[rgb_image, text],
                image_shapes=rgb_image.size,
                num_timesteps=num_timesteps,
            )[
                -1
            ]  # PIL

            if dataset_name == "detection":
                # 你已有的 TP/FP 统计（返回形如 (N,) 的 0/1）
                tp, fp = tpfp(gt_bboxes, pred_img, gen_image, cls_name)

                # 保证是可序列化的一维 int 列表
                tp = np.asarray(tp).reshape(-1).astype(int).tolist()
                fp = np.asarray(fp).reshape(-1).astype(int).tolist()

                tp_dict[cls_name].extend(tp)  # 累加 0/1
                fp_dict[cls_name].extend(fp)  # 注意用 extend 而不是 append
            elif dataset_name == "ade20k":
                rgb_mse_list.append(rgb_mse(pred_img, gen_image))
                rgb_delta1_acc_list.append(rgb_delta1_acc(pred_img, gen_image))
                rgb_absrel_list.append(rgb_absrel(pred_img, gen_image))

            # ===== 可选保存可视化 =====
            if save_to_dir is not None:
                curr_dir = os.path.join(
                    save_to_dir, f"step_{curr_step}", dataset_name, data_id
                )
                os.makedirs(curr_dir, exist_ok=True)
                rgb_image.save(os.path.join(curr_dir, "rgb.png"))
                gen_image.save(os.path.join(curr_dir, f"{dataset_name}_gt.png"))
                pred_img.save(os.path.join(curr_dir, f"{dataset_name}_pred.png"))

    # ========== 分布式聚合到 rank 0 ==========
    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0

    if dataset_name == "ade20k":
        # 把本 rank 的结果打包
        payload = {
            "rgb_mse_list": rgb_mse_list,
            "rgb_delta1_acc_list": rgb_delta1_acc_list,
            "rgb_absrel_list": rgb_absrel_list,
        }

        if world_size > 1:
            if rank == 0:
                gathered = [None] * world_size
                dist.gather_object(payload, gathered, dst=0)
                agg_rgb_mse_list = []
                agg_rgb_delta1_acc_list = []
                agg_rgb_absrel_list = []
                # 合并所有 rank 的字典
                for p in gathered:
                    agg_rgb_mse_list.extend(p["rgb_mse_list"])
                    agg_rgb_delta1_acc_list.extend(p["rgb_delta1_acc_list"])
                    agg_rgb_absrel_list.extend(p["rgb_absrel_list"])
            else:
                dist.gather_object(payload, dst=0)
                # 非 0 卡不做后续计算，也不写文件
                return None
        else:
            agg_rgb_mse_list = rgb_mse_list
            agg_rgb_delta1_acc_list = rgb_delta1_acc_list
            agg_rgb_absrel_list = rgb_absrel_list

        mean_rgb_mse = np.mean(agg_rgb_mse_list)
        mean_rgb_delta1_acc = np.mean(agg_rgb_delta1_acc_list)
        mean_rgb_absrel = np.mean(agg_rgb_absrel_list)

        # 组织写入内容
        step_key = f"step_{curr_step}"
        to_write = {
            "mean_rgb_mse": float(mean_rgb_mse),
            "mean_rgb_delta1_acc": float(mean_rgb_delta1_acc),
            "mean_rgb_absrel": float(mean_rgb_absrel),
        }
    elif dataset_name == "detection":
        # 把本 rank 的结果打包
        payload = {"tp": tp_dict, "fp": fp_dict}

        if world_size > 1:
            if rank == 0:
                gathered = [None] * world_size
                dist.gather_object(payload, gathered, dst=0)
                agg_tp, agg_fp = {}, {}
                # 合并所有 rank 的字典
                for p in gathered:
                    for k, v in p["tp"].items():
                        agg_tp.setdefault(k, []).extend(v)
                    for k, v in p["fp"].items():
                        agg_fp.setdefault(k, []).extend(v)
            else:
                dist.gather_object(payload, dst=0)
                # 非 0 卡不做后续计算，也不写文件
                return None
        else:
            agg_tp, agg_fp = tp_dict, fp_dict

        # ========== rank 0 计算 AP 与 mAP，写入 JSON ==========
        # 期望 cal_ap 返回 {cls: ap, ...} 或 (per_class_dict, mAP)
        ap_result = cal_ap(agg_tp, agg_fp)

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
        step_key = f"step_{curr_step}"
        to_write = {
            "ap_per_class": {str(k): float(v) for k, v in ap_per_class.items()},
            "mAP": float(mean_ap),
        }

    if save_to_dir is not None:
        os.makedirs(save_to_dir, exist_ok=True)
        metrics_path = os.path.join(save_to_dir, "metric.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_all = json.load(f)
        else:
            metrics_all = {}

        if step_key not in metrics_all.keys():
            metrics_all[step_key] = {dataset_name: to_write}
        else:
            metrics_all[step_key] = {dataset_name: to_write}
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_all, f, ensure_ascii=False, indent=2)

    return to_write


def main():
    # 1. 初始化分布式环境和设备
    assert torch.cuda.is_available()
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # 2. 解析参数
    parser = HfArgumentParser(EvalArguments)
    (eval_args,) = parser.parse_args_into_dataclasses()
    # 5. 设置随机种子
    seed = eval_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # 6. 准备模型（语言模型、视觉模型等）
    llm_config = Qwen2Config.from_json_file(
        os.path.join(eval_args.model_path, "llm_config.json")
    )
    llm_config.freeze_und = True
    llm_config.gradient_checkpointing = False

    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(eval_args.model_path, "vit_config.json")
    )
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    vae_model, vae_config = load_ae(
        local_path=(os.path.join(eval_args.model_path, "ae.safetensors"))
    )
    vae_model = vae_model.to(dtype=torch.bfloat16).to(device)

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        max_latent_size=64,
        timestep_shift=50,
    )

    # 7. 设置Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(eval_args.model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    valid_loader = None
    metric_funcs = [getattr(metric, _met) for _met in eval_args.valid_metrics]
    val_metrics = MetricTracker(*[m.__name__ for m in metric_funcs])
    depth_transform = DepthImageTransform(
        image_stride=16,
        max_image_size=1024,
        min_image_size=512,
    )
    vit_transform = ImageTransform(
        image_stride=14,
        max_image_size=518,
        min_image_size=224,
    )
    if eval_args.valid_dataset_name in ["vkitti", "kitti", "nyu", "hypersim"]:
        valid_dataset = DepthInferDataset(
            transform=depth_transform,
            image_dir=eval_args.valid_image_path,
            jsonl_path=eval_args.valid_data_path,
            dataset_name=eval_args.valid_dataset_name,
        )
        valid_sampler = DistributedSampler(
            valid_dataset,
            seed=eval_args.data_seed,
            shuffle=False,  # 推理时不打乱数据，确保可复现
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,
            sampler=valid_sampler,
            num_workers=eval_args.num_workers,
            collate_fn=collate_fn_depth,
            drop_last=False,
            prefetch_factor=eval_args.prefetch_factor,
            pin_memory=True,
        )
    elif eval_args.valid_dataset_name in ["detection"]:
        valid_dataset = DetectionInferDataset(
            transform=depth_transform,
            image_dir=eval_args.valid_image_path,
            jsonl_path=eval_args.valid_data_path,
            dataset_name=eval_args.valid_dataset_name,
        )
        valid_sampler = DistributedSampler(
            valid_dataset,
            seed=eval_args.data_seed,
            shuffle=False,  # 推理时不打乱数据，确保可复现
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,
            sampler=valid_sampler,
            num_workers=eval_args.num_workers,
            collate_fn=collate_fn_detection,
            drop_last=False,
            prefetch_factor=eval_args.prefetch_factor,
            pin_memory=True,
        )
    elif eval_args.valid_dataset_name in ["ade20k"]:
        valid_dataset = SegmentationInferDataset(
            transform=depth_transform,
            image_dir=eval_args.valid_image_path,
            jsonl_path=eval_args.valid_data_path,
            dataset_name=eval_args.valid_dataset_name,
        )
        valid_sampler = DistributedSampler(
            valid_dataset,
            seed=eval_args.data_seed,
            shuffle=False,  # 推理时不打乱数据，确保可复现
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,
            sampler=valid_sampler,
            num_workers=eval_args.num_workers,
            collate_fn=collate_fn_segmentation,
            drop_last=False,
            prefetch_factor=eval_args.prefetch_factor,
            pin_memory=True,
        )

    curr_ckpt_dir = os.path.join(eval_args.model_path_param, "ema.safetensors")

    with init_empty_weights():
        language_model_eval = Qwen2ForCausalLM(llm_config)
        vit_model_eval = SiglipVisionModel(vit_config)
        model_eval = Bagel(language_model_eval, vit_model_eval, config)
        model_eval.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )

    # state_dict = st.load_file("ema.safetensors")
    # model_eval.load_state_dict(state_dict, strict=False)  # 或按需处理 key
    model_eval = load_checkpoint_and_dispatch(
        model_eval,
        checkpoint=curr_ckpt_dir,
        device_map={"": device},
        offload_buffers=False,
        dtype=torch.bfloat16,
    )
    model_eval = model_eval.eval()

    inferencer = InterleaveInferencer(
        model=model_eval,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vit_transform=vit_transform,
        vae_transform=depth_transform,
        new_token_ids=new_token_ids,
        device=device,
    )
    if eval_args.valid_dataset_name in ["vkitti", "kitti", "nyu", "hypersim"]:
        print("Validating on depth dataset...")
        validate(
            inferencer=inferencer,
            valid_loader=valid_loader,
            metric_funcs=metric_funcs,
            alignment=eval_args.valid_alignment,
            metric_tracker=val_metrics,
            curr_step=eval_args.model_path_param.split("/")[-1],
            dataset_name=eval_args.valid_dataset_name,
            transform=depth_transform,
            num_timesteps=eval_args.num_timesteps,
            save_to_dir=eval_args.save_to_dir,
        )
    elif eval_args.valid_dataset_name in ["detection", "ade20k"]:
        validate_detection(
            inferencer=inferencer,
            valid_loader=valid_loader,
            curr_step=eval_args.model_path_param.split("/")[-1],
            num_timesteps=eval_args.num_timesteps,
            save_to_dir=eval_args.save_to_dir,
            dataset_name=eval_args.valid_dataset_name,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
