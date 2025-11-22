# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import shutil
import time
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    ShardedStateDictConfig,
    LocalStateDictConfig,
    _state_dict_utils,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.bagel.modeling_utils import (
    MLPconnector,
    TimestepEmbedder,
    PositionEmbedding,
)
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer,
    Qwen2MoEDecoderLayer,
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer
from torch.distributed.checkpoint import (
    save,
    load,
    FileSystemWriter,
    FileSystemReader,
)


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy,
        backward_prefetch,
        cpu_offload,
        num_replicate,
        use_orig_params,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard
        self.use_orig_params = use_orig_params


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == "HYBRID_SHARD":
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard"),
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=fsdp_config.use_orig_params,
    )


def _to_bf16(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    仅把浮点张量转换为 bfloat16；其余 dtype（int/bool 等）保持不变。
    """
    return {
        name: (
            tensor
            if tensor.dtype == torch.bfloat16 or not torch.is_floating_point(tensor)
            else tensor.to(torch.bfloat16)
        )
        for name, tensor in state_dict.items()
    }


def _get_cfg_attr(cfg, key, default=None):
    # 兼容 Namespace / dataclass / dict
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir,
        train_steps,
        model,
        ema_model,
        optimizer,
        scheduler,
        # data_status,
        logger,
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")
        torch.cuda.empty_cache()

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                ema_state_dict = _to_bf16(ema_state_dict)
                if dist.get_rank() == 0:
                    save_file(
                        ema_state_dict, os.path.join(save_path, "ema.safetensors")
                    )
                del ema_state_dict
                torch.cuda.empty_cache()

        # with FSDP.state_dict_type(
        #     model,
        #     StateDictType.FULL_STATE_DICT,
        #     FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        # ):
        #     model_state_dict = _to_bf16(model.state_dict())
        #     if dist.get_rank() == 0:
        #         save_file(
        #             model_state_dict, os.path.join(save_path, "model.safetensors")
        #         )
        #     del model_state_dict
        #     torch.cuda.empty_cache()

        # with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        #     if fsdp_config.sharding_strategy == "FULL_SHARD":
        #         shard_index = dist.get_rank()
        #         total_shards = dist.get_world_size()
        #     elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
        #         shard_index = dist.get_rank() % fsdp_config.num_shard
        #         total_shards = fsdp_config.num_shard
        #     else:
        #         raise NotImplementedError

        #     optimizer_save_path = os.path.join(
        #         save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
        #     )
        #     if fsdp_config.sharding_strategy == "FULL_SHARD":
        #         torch.save(optimizer.state_dict(), optimizer_save_path)
        #     elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
        #         if dist.get_rank() < fsdp_config.num_shard:
        #             torch.save(optimizer.state_dict(), optimizer_save_path)
        #     else:
        #         raise NotImplementedError

        # if dist.get_rank() == 0 and scheduler is not None:
        #     torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        # if dist.get_rank() == 0 and data_status is not None:
        #     torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def fsdp_save_ckpt_sharded(
        ckpt_dir,
        train_steps,
        model,
        ema_model,
        optimizer,
        scheduler,
        data_status,
        logger,
        fsdp_config,
    ):
        rank = dist.get_rank()
        ws = dist.get_world_size()

        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        shards_dir = os.path.join(save_path, "_shards")
        if rank == 0:
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Saving checkpoint (sharded→CPU聚合) to {save_path}.")

        # -------- 1) 分片保存（显存最低，不做 FULL 聚合） --------
        dist.barrier()

        # 建议在调用本函数前外层已 eval()/no_grad() + zero_grad()
        cfg = ShardedStateDictConfig(offload_to_cpu=True)

        # 模型分片
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
            model_sd_sharded = model.state_dict()
        payload = {"model": model_sd_sharded}

        # EMA 分片（如有）
        if ema_model is not None:
            with FSDP.state_dict_type(ema_model, StateDictType.SHARDED_STATE_DICT, cfg):
                ema_sd_sharded = ema_model.state_dict()
            payload["ema"] = ema_sd_sharded

        # 写分片到本步目录下的 _shards/
        save(payload, storage_writer=FileSystemWriter(shards_dir))
        dist.barrier()

        # -------- 2) 仅 rank0：在 CPU 聚合为完整权重并写成 safetensors --------
        if rank == 0:
            agg_t0 = time.time()
            reader = FileSystemReader(shards_dir)

            # 从 fsdp_config 获取 CPU 模型构造函数
            cpu_model_ctor = _get_cfg_attr(fsdp_config, "cpu_model_ctor", None)
            if cpu_model_ctor is None or not callable(cpu_model_ctor):
                raise RuntimeError(
                    "fsdp_config.cpu_model_ctor 未提供或不可调用：需要一个返回未包FSDP且同架构模型的构造函数。"
                )

            # 2a) 模型：在 CPU 聚合
            cpu_model = cpu_model_ctor().cpu()
            cpu_model.eval()
            tgt = {"model": cpu_model.state_dict()}  # 目标容器：完整 state_dict（CPU）
            load(tgt, storage_reader=reader)  # DCP 负责把分片重组成完整权重
            full_model_sd = _to_bf16(tgt["model"])
            save_file(full_model_sd, os.path.join(save_path, "model.safetensors"))

            # 2b) EMA：如存在，则同样聚合
            if "ema" in payload:
                cpu_ema_ctor = _get_cfg_attr(
                    fsdp_config, "cpu_ema_model_ctor", cpu_model_ctor
                )
                cpu_ema = cpu_ema_ctor().cpu()
                cpu_ema.eval()
                tgt_ema = {"ema": cpu_ema.state_dict()}
                # 若不存在ema分片，load_state_dict会抛错；做一次保护
                try:
                    load(tgt_ema, storage_reader=reader)
                    full_ema_sd = _to_bf16(tgt_ema["ema"])
                    save_file(full_ema_sd, os.path.join(save_path, "ema.safetensors"))
                except Exception as e:
                    logger.warning(f"EMA aggregation skipped or partial: {e}")

            # 2c) 删除分片文件夹
            try:
                shutil.rmtree(shards_dir)
            except Exception as e:
                logger.warning(f"Failed to remove shards dir {shards_dir}: {e}")

            logger.info(
                f"[aggregate] CPU合并并写safetensors耗时: {time.time() - agg_t0:.3f}s"
            )

        dist.barrier()
        # 到这里，save_path 下只剩下 model.safetensors / ema.safetensors（如有）
        # 外层已有总耗时打印的话，这里不重复打印
        return

    @staticmethod
    def try_load_ckpt(
        resume_from, logger, model, ema_model=None, resume_from_ema=False
    ):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            model_state_dict.pop("latent_pos_embed.pos_embed")
            model_state_dict.pop("vit_pos_embed.pos_embed")
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                ema_state_dict.pop("latent_pos_embed.pos_embed")
                ema_state_dict.pop("vit_pos_embed.pos_embed")
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(
                optimizer_state_dict_path, map_location="cpu", weights_only=True
            )
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(
                scheduler_state_dict_path, weights_only=True, map_location="cpu"
            )
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(
                    data_status_path, weights_only=True, map_location="cpu"
                )
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer,
        SiglipEncoderLayer,
        MLPconnector,
        Qwen2MoEDecoderLayer,
        Qwen2MoTDecoderLayer,
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(
                new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype)
            )

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
