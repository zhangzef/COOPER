# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from PIL import Image
import torch
from .data_utils import (
    get_flattened_position_ids_extrapolate,
    len2weight,
    patchify,
    prepare_attention_mask_per_sample,
)

REFLECTION_PROMPT = "\nHere is the result of the depth-estimation/segmentation. Please note that the result of the depth-estimation/segmentation is not always accurate. Please check it carefully. \nNow please continue to think in <think>...</think> and then decide whether to continue to generate the depth-estimation/segmentation in <depth-estimation>...</depth-estimation>/<segmentation>...</segmentation> or give the answer in <answer>...</answer>.\n"


class DataConfig:
    def __init__(
        self,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    ):
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size


class OutputTransfer:
    def __init__(
        self,
        tokenizer,
        transform,
        vit_transform,
        data_config,
        max_num_tokens,
        new_token_ids,
        use_flex=False,
    ):
        self.tokenizer = tokenizer
        self.transform = transform
        self.vit_transform = vit_transform
        self.data_config = data_config
        self.use_flex = use_flex
        self.max_num_tokens = max_num_tokens
        for k, v in new_token_ids.items():
            setattr(self, k, v)

    def __call__(self, output_list, device, id_list):
        sequence_status_list = []
        for i, output in enumerate(output_list):
            self.format_check(output)
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

            data = self._init_data()
            data = self._add_text(data, output[0], need_loss=False)
            data = self._add_image(
                data,
                self.pil_img2rgb(output[1]),
                need_vae=True,
                need_vit=True,
            )
            data = self._add_text(data, output[2], need_loss=False)

            # completions_tokens_text = 0
            prompt_tokens = data["num_tokens"]
            gen_image_tokens = 0
            output = output[3:]
            for j in range(0, len(output)):
                if type(output[j]) is str:
                    # before = data["num_tokens"]
                    if output[j] == REFLECTION_PROMPT:
                        data = self._add_text(data, output[j], need_loss=False)
                    else:
                        data = self._add_text(data, output[j], need_loss=True)
                    # completions_tokens_text += data["num_tokens"] - before
                else:
                    pre_num_tokens = data["num_tokens"]
                    data = self._add_image(
                        data,
                        self.pil_img2rgb(output[j]),
                        need_vae=True,
                        need_vit=True,
                    )
                    gen_image_tokens += data["num_tokens"] - pre_num_tokens
            data["completions_tokens"] = data["num_tokens"] - prompt_tokens
            # data["completions_tokens_text"] = completions_tokens_text
            sequence_status = self.set_sequence_status()
            sequence_status = self.pack_sequence(data, sequence_status)
            # print(f"ce_loss_indexes:{sequence_status['ce_loss_indexes']}")
            sequence_status["completions_tokens_text"] = len(
                sequence_status["ce_loss_indexes"]
            )
            sequence_status["gen_image_tokens"] = gen_image_tokens
            sequence_status["data_idx"] = id_list[i]
            sequence_status_list.append(
                self.cuda(self.to_tensor(sequence_status), device)
            )
        return sequence_status_list

    def set_sequence_status(self):
        sequence_status = dict(
            # 每个样本的长度列表，记录每个样本在打包后序列中的总token数
            # 数据类型: int
            sequence_length=0,
            # 每个样本补全部分的token总数（包含图像和文本）
            # 数据类型: int
            completions_tokens=0,
            # 每个样本补全部分的文本token数量
            # 数据类型: int
            completions_tokens_text=0,
            # 打包后序列中所有token的位置ID（用于RoPE位置编码）
            # 格式: [pos_id_0, pos_id_1, ..., pos_id_n]
            # 数据类型: List[int]
            packed_position_ids=list(),
            # 嵌套注意力掩码列表，每个元素是一个样本的注意力掩码张量
            # 格式: [mask_tensor_0, mask_tensor_1, ...]
            # 数据类型: List[torch.Tensor]
            nested_attention_masks=list(),
            # 每个片段的长度列表（用于注意力机制）
            # 格式: [len_0, len_1, len_2, ...] 其中每个len表示一个文本/图像片段的token数量
            # 数据类型: List[int]
            split_lens=list(),
            # 注意力模式列表，对应split_lens中的每个片段
            # 值为"causal"（因果注意力）或"full"（全连接注意力）
            # 格式: ["causal", "full", "causal", ...]
            # 数据类型: List[str]
            attn_modes=list(),
            # 打包后的所有文本token ID序列
            # 格式: [token_id_0, token_id_1, ..., token_id_n]
            # 数据类型: List[int]
            packed_text_ids=list(),
            # 文本token在打包后序列中的位置索引
            # 格式: [index_0, index_1, ..., index_m] 其中index表示packed_text_ids中token在完整序列中的位置
            # 数据类型: List[int]
            packed_text_indexes=list(),
            # 打包后的所有文本token ID序列
            # 格式: [token_id_0, token_id_1, ..., token_id_n]
            # 数据类型: List[int]
            ce_loss_text_ids=list(),
            # 用于计算交叉熵损失的token索引
            # 格式: [loss_index_0, loss_index_1, ...]
            # 数据类型: List[int]
            ce_loss_indexes=list(),
            # VAE图像张量列表（原始图像经过VAE编码后的潜在表示）
            # 格式: [vae_tensor_0, vae_tensor_1, ...] 其中每个tensor形状为[C, H, W]
            # 数据类型: List[torch.Tensor]
            vae_image_tensors=list(),
            # VAE图像token的位置ID列表
            # 格式: [[pos_id_00, pos_id_01, ...], [pos_id_10, ...], ...]
            # 每个子列表对应一个图像的位置ID
            # 数据类型: List[List[int]] (后续会被转换为torch.Tensor)
            packed_latent_position_ids=list(),
            # VAE潜在空间的形状列表
            # 格式: [(h0, w0), (h1, w1), ...] 每个元组表示一个图像潜在表示的高度和宽度
            # 数据类型: List[Tuple[int, int]]
            vae_latent_shapes=list(),
            # VAE图像token在打包后序列中的位置索引
            # 格式: [index_0, index_1, ..., index_k]
            # 数据类型: List[int] (后续会被转换为torch.Tensor)
            packed_vae_token_indexes=list(),
            # 时间步列表（用于扩散模型等）
            # 格式: [timestep_0, timestep_1, ...] 通常为float值，可能包含-inf
            # 数据类型: List[float] (后续会被转换为torch.Tensor)
            packed_timesteps=list(),
            # ViT图像token列表
            # 格式: [vit_token_tensor_0, vit_token_tensor_1, ...] 每个tensor形状为[num_patches, dim]
            # 数据类型: List[torch.Tensor] (后续会被concat为单个torch.Tensor)
            packed_vit_tokens=list(),
            # 每个ViT图像的token序列长度
            # 格式: [len_0, len_1, ...] 每个值表示对应图像的patch数量
            # 数据类型: List[int] (后续会被转换为torch.Tensor)
            vit_token_seqlens=list(),
            # ViT图像token的位置ID列表
            # 格式: [[pos_id_00, pos_id_01, ...], [pos_id_10, ...], ...]
            # 每个子列表对应一个图像的位置ID
            # 数据类型: List[List[int]] (后续会被转换为torch.Tensor)
            packed_vit_position_ids=list(),
            # ViT图像token在打包后序列中的位置索引
            # 格式: [index_0, index_1, ..., index_p]
            # 数据类型: List[int] (后续会被转换为torch.Tensor)
            packed_vit_token_indexes=list(),
        )
        return sequence_status

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample["image_tensor_list"]
        text_ids_list = sample["text_ids_list"]
        sequence_plan = sample["sequence_plan"]

        split_lens, attn_modes = list(), list()
        curr = 0
        curr_rope_id = 0

        for item in sequence_plan:
            curr_split_len = 0
            if item["type"] == "text":
                text_ids = text_ids_list.pop(0)

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status["packed_text_ids"].extend(shifted_text_ids)
                sequence_status["packed_text_indexes"].extend(
                    range(curr, curr + len(shifted_text_ids))
                )
                if item["loss"] == 1:
                    sequence_status["ce_loss_indexes"].extend(
                        range(curr, curr + len(shifted_text_ids))
                    )
                    sequence_status["ce_loss_text_ids"].extend(shifted_text_ids)
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status["packed_text_ids"].append(self.eos_token_id)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:  # <|im_end|> may have loss
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_text_ids"].append(self.eos_token_id)
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status["packed_position_ids"].extend(
                    range(curr_rope_id, curr_rope_id + curr_split_len)
                )
                curr_rope_id += curr_split_len

            elif item["type"] == "vit_image":
                image_tensor = image_tensor_list.pop(0)

                # add a <|startofimage|> token
                sequence_status["packed_text_ids"].append(self.start_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status["packed_vit_token_indexes"].extend(
                    range(curr, curr + num_img_tokens)
                )
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status["packed_vit_tokens"].append(vit_tokens)
                sequence_status["vit_token_seqlens"].append(num_img_tokens)
                sequence_status["packed_vit_position_ids"].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1),
                        image_tensor.size(2),
                        self.data_config.vit_patch_size,
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side,
                    )
                )

                # add a <|endofimage|> token
                sequence_status["packed_text_ids"].append(self.end_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:  # <|endofimage|> may have loss
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_text_ids"].append(self.end_of_image)
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status["packed_position_ids"].extend(
                    [curr_rope_id] * curr_split_len
                )
                curr_rope_id += 1

            elif item["type"] == "vae_image":
                image_tensor = image_tensor_list.pop(0)

                # add a <|startofimage|> token
                sequence_status["packed_text_ids"].append(self.start_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                sequence_status["vae_image_tensors"].append(image_tensor)
                sequence_status["packed_latent_position_ids"].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1),
                        image_tensor.size(2),
                        self.data_config.vae_image_downsample,
                        max_num_patches_per_side=self.data_config.max_latent_size,
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status["vae_latent_shapes"].append((h, w))

                num_img_tokens = w * h
                sequence_status["packed_vae_token_indexes"].extend(
                    range(curr, curr + num_img_tokens)
                )
                timestep = float("-inf")

                sequence_status["packed_timesteps"].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                # add a <|endofimage|> token
                sequence_status["packed_text_ids"].append(self.end_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                # <|endofimage|> may have loss
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_text_ids"].append(self.end_of_image)
                curr += 1
                curr_split_len += 1
                sequence_status["packed_position_ids"].extend(
                    [curr_rope_id] * (num_img_tokens + 2)
                )
                if item["loss"] == 0:
                    curr_rope_id += 1

            split_lens.append(curr_split_len)

        sequence_status["sequence_length"] = curr
        sequence_status["completions_tokens"] = sample["completions_tokens"]
        # sequence_status["completions_tokens_text"] = sample["completions_tokens_text"]
        # prepare attention mask
        if not self.use_flex:
            sequence_status["nested_attention_masks"].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status["split_lens"].extend(split_lens)
            sequence_status["attn_modes"].extend(attn_modes)

        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sequence_status["sequence_length"],
            packed_text_ids=torch.tensor(sequence_status["packed_text_ids"]),
            packed_text_indexes=torch.tensor(sequence_status["packed_text_indexes"]),
            packed_position_ids=torch.tensor(sequence_status["packed_position_ids"]),
            ce_loss_text_ids=torch.tensor(sequence_status["ce_loss_text_ids"]),
            ce_loss_indexes=torch.tensor(sequence_status["ce_loss_indexes"]),
            completions_tokens=sequence_status["completions_tokens"],
            completions_tokens_text=sequence_status["completions_tokens_text"],
            gen_image_tokens=sequence_status["gen_image_tokens"],
            data_idx=sequence_status["data_idx"],
        )
        if not self.use_flex:
            data["nested_attention_masks"] = sequence_status["nested_attention_masks"]
        else:
            sequence_len = sequence_status["sequence_length"]
            pad_len = self.max_num_tokens - sequence_len
            data["split_lens"] = sequence_status["split_lens"] + [pad_len]
            data["attn_modes"] = sequence_status["attn_modes"] + ["causal"]
            data["sequence_length"] += pad_len

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status["vae_image_tensors"]) > 0:
            image_tensors = sequence_status.pop("vae_image_tensors")
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[
                    i, :, : image_tensor.shape[1], : image_tensor.shape[2]
                ] = image_tensor

            data["padded_images"] = padded_images
            data["patchified_vae_latent_shapes"] = sequence_status["vae_latent_shapes"]
            data["packed_latent_position_ids"] = torch.cat(
                sequence_status["packed_latent_position_ids"], dim=0
            )
            data["packed_vae_token_indexes"] = torch.tensor(
                sequence_status["packed_vae_token_indexes"]
            )

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status["packed_vit_tokens"]) > 0:
            data["packed_vit_tokens"] = torch.cat(
                sequence_status["packed_vit_tokens"], dim=0
            )
            data["packed_vit_position_ids"] = torch.cat(
                sequence_status["packed_vit_position_ids"], dim=0
            )
            data["packed_vit_token_indexes"] = torch.tensor(
                sequence_status["packed_vit_token_indexes"]
            )
            data["vit_token_seqlens"] = torch.tensor(
                sequence_status["vit_token_seqlens"]
            )

        # if the model is required to perform visual generation
        if len(sequence_status["packed_timesteps"]) > 0:
            data["packed_timesteps"] = torch.tensor(sequence_status["packed_timesteps"])

        return data

    def cuda(self, data, device):
        data["packed_text_ids"] = data["packed_text_ids"].to(device)
        data["packed_text_indexes"] = data["packed_text_indexes"].to(device)
        data["packed_position_ids"] = data["packed_position_ids"].to(device)
        data["ce_loss_indexes"] = data["ce_loss_indexes"].to(device)
        data["ce_loss_text_ids"] = data["ce_loss_text_ids"].to(device)
        data["completions_tokens"] = data["completions_tokens"]
        data["completions_tokens_text"] = data["completions_tokens_text"]
        data["gen_image_tokens"] = data["gen_image_tokens"]
        data["data_idx"] = data["data_idx"]

        if not self.use_flex:
            data["nested_attention_masks"] = [
                item.to(device) for item in data["nested_attention_masks"]
            ]

        if "padded_images" in data.keys():
            data["padded_images"] = data["padded_images"].to(device)
            data["packed_vae_token_indexes"] = data["packed_vae_token_indexes"].to(
                device
            )
            data["packed_latent_position_ids"] = data["packed_latent_position_ids"].to(
                device
            )

        if "packed_timesteps" in data.keys():
            data["packed_timesteps"] = data["packed_timesteps"].to(device)

        if "packed_vit_tokens" in data.keys():
            data["packed_vit_tokens"] = data["packed_vit_tokens"].to(device)
            data["packed_vit_position_ids"] = data["packed_vit_position_ids"].to(device)
            data["packed_vit_token_indexes"] = data["packed_vit_token_indexes"].to(
                device
            )
            data["vit_token_seqlens"] = data["vit_token_seqlens"].to(device)

        return data

    def format_check(self, output):
        assert (
            len(output) >= 4
        ), "the length of the output list should be at least three"
        assert isinstance(
            output[0], str
        ), "the first element of the output list should be a string"
        assert isinstance(
            output[1], Image.Image
        ), "the second element of the output list should be an image"
        assert isinstance(
            output[2], str
        ), "the third element of the output list should be a string"
        assert isinstance(
            output[-1], str
        ), "the last element of the output list should be a string"

    def pil_img2rgb(self, image):
        if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
            image = image.convert("RGBA")
            white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
            white.paste(image, mask=image.split()[3])
            image = white
        else:
            image = image.convert("RGB")

        return image

    def _init_data(self):
        data = {
            "sequence_plan": [],
            "text_ids_list": [],
            "image_tensor_list": [],
            "num_tokens": 0,
        }
        return data

    def _add_text(self, data, text, need_loss):
        text_ids = self.tokenizer.encode(text)
        data["num_tokens"] += len(text_ids)
        data["text_ids_list"].append(text_ids)
        data["sequence_plan"].append(
            {
                "type": "text",
                "loss": int(need_loss),
                "special_token_loss": int(need_loss),
                "special_token_label": None,
            }
        )
        return data

    def _add_image(self, data, image, need_vae, need_vit):
        assert need_vae or need_vit

        if need_vae:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )

            image_tensor = self.transform(image, is_rgb=True)
            height, width = image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.transform.stride**2
            data["image_tensor_list"].append(image_tensor.clone())

        if need_vit:
            data["sequence_plan"].append(
                {
                    "type": "vit_image",
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
