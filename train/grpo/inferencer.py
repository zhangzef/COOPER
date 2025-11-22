# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any
import io, base64
from PIL import Image
import torch
from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache
import re
import numpy as np
import matplotlib

VLM_THINK_SYSTEM_PROMPT = """You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"""

GEN_THINK_SYSTEM_PROMPT = """You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here"""

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

IMAGE_REASON_SYSTEM_PROMPT = """You are a specialized multimodal assistant. Your aim is to answer the questions given by users based on the depth estimation and semantic segmentation you have generated.

# Instruction
Please don't thinking and just give the final answer directly in <answer> tag. Besides, you also need to put a simple and direct answer in \\boxed{{}} for verification.

The structure of your response should be like this:
<answer> ... </answer>
"""

DEPTH_PROMPT = "<depth-estimation>Estimate the depth of the image and generate the depth map.</depth-estimation>"

SEGMENTATION_PROMPT = "<segmentation>Segment the objects in the image with different colors.</segmentation>"

REFLECTION_PROMPT = "\nHere is the result of the depth-estimation/segmentation. Please note that the result of the depth-estimation/segmentation is not always accurate. Please check it carefully. \nNow please continue to think in <think>...</think> and then decide whether to continue to generate the depth-estimation/segmentation in <depth-estimation>...</depth-estimation>/<segmentation>...</segmentation> or give the answer in <answer>...</answer>.\n"


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)  # 把图像写入内存缓冲区
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64


def dict_to_device(dict: Dict, device):
    for key in dict:
        if isinstance(dict[key], torch.Tensor):
            dict[key] = dict[key].to(device)
    return dict


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    else:
        raise TypeError("img should be np.ndarray or torch.Tensor")
    return hwc


class InterleaveInferencer:
    def __init__(
        self,
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids,
        device,
    ):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.device = device

    def init_gen_context(self):
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            ),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference,

        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = dict_to_device(generation_input, self.device)

        past_key_values = self.model.forward_cache_update_text(
            past_key_values, **generation_input
        )
        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values

        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference,

        assert vae or vit
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = dict_to_device(generation_input, self.device)
            past_key_values = self.model.forward_cache_update_vae(
                self.vae_model, past_key_values, **generation_input
            )

        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            generation_input = dict_to_device(generation_input, self.device)
            past_key_values = self.model.forward_cache_update_vit(
                past_key_values, **generation_input
            )

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values

        return gen_context

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0,
        out_depth=False,
    ):
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        generation_input = dict_to_device(generation_input, self.device)

        # text cfg
        cfg_text_past_key_values = cfg_text_precontext["past_key_values"]
        kv_lens_cfg = cfg_text_precontext["kv_lens"]
        ropes_cfg = cfg_text_precontext["ropes"]
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )
        generation_input_cfg_text = dict_to_device(
            generation_input_cfg_text, self.device
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext["past_key_values"]
        kv_lens_cfg = cfg_img_precontext["kv_lens"]
        ropes_cfg = cfg_img_precontext["ropes"]
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )
        generation_input_cfg_img = dict_to_device(generation_input_cfg_img, self.device)

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text[
                "cfg_packed_position_ids"
            ],
            cfg_text_packed_query_indexes=generation_input_cfg_text[
                "cfg_packed_query_indexes"
            ],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text[
                "cfg_packed_key_value_indexes"
            ],
            cfg_img_packed_position_ids=generation_input_cfg_img[
                "cfg_packed_position_ids"
            ],
            cfg_img_packed_query_indexes=generation_input_cfg_img[
                "cfg_packed_query_indexes"
            ],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img[
                "cfg_packed_key_value_indexes"
            ],
        )

        if not out_depth:
            image = self.decode_image(unpacked_latent[0], image_shape)
            return image
        else:
            image = self.decode_depth(unpacked_latent[0], image_shape)
            return image

    def decode_depth(self, latent, image_shape):
        # decode latent to depth map
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(
            1,
            h,
            w,
            self.model.latent_patch_size,
            self.model.latent_patch_size,
            self.model.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1,
            self.model.latent_channel,
            h * self.model.latent_patch_size,
            w * self.model.latent_patch_size,
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).mean(dim=-1)
        return image

    def decode_image(self, latent, image_shape):
        # decode latent to image
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(
            1,
            h,
            w,
            self.model.latent_patch_size,
            self.model.latent_patch_size,
            self.model.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1,
            self.model.latent_channel,
            h * self.model.latent_patch_size,
            w * self.model.latent_patch_size,
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(
        self,
        gen_context,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input = self.model.prepare_start_tokens(
            kv_lens, ropes, self.new_token_ids
        )
        generation_input = dict_to_device(generation_input, self.device)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            end_token_id=self.new_token_ids["eos_token_id"],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:, 0])
        output = output.split("<|im_end|>")[0].split("<|im_start|>")[1]
        return output

    @torch.no_grad()
    def depth_map_generation(
        self,
        input_lists: List[Union[str, Image.Image]],
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        # generate the depth map for evaluation, not generate the visualization image

        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(
                        input_term, cfg_img_context
                    )

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=True
                    )

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            return self.gen_image(
                image_shapes,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                out_depth=True,
            )

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        # origin BAGEL interleave inference function

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(
                    system_prompt, cfg_img_context
                )

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(
                        input_term, cfg_img_context
                    )

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=not understanding_output
                    )

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                    )
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )

                output_list.append(img)

        return output_list

    @torch.no_grad()
    def image_generation_edit(
        self,
        input_list: List[Union[str, Image.Image]],
        max_think_token_n=1024,
        do_sample=False,
        think=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        # image edit function, you can use it to generate the segmentation map or the netural image
        # the input_list should have the input image and the text prompt

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(
                    system_prompt, cfg_img_context
                )

            for input_term in input_list:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(
                        input_term, cfg_img_context
                    )

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=True
                    )
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if think:
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                gen_context = self.update_context_text(gen_text, gen_context)
                output_list.append(gen_text)

            img = self.gen_image(
                image_shapes,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
            )

            output_list.append(img)

        return output_list

    @torch.no_grad()
    def interleave_reason_tool_condition(
        self,
        input_lists: List[Union[str, Image.Image]],
        max_inter_num=3,
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=30,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        top_p=1.0,
        **kwargs,
    ) -> List[Union[str, Image.Image]]:
        # cooperative reasoning and perception generation function
        # the input_list shuould have the input image and the text prompt
        # it can generate the interleaved multimodal chain-of-thought by the model
        # but the image generation is decided by the model itself

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        edit_cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            system_prompt = INTERLEAVE_REASON_SYSTEM_PROMPT
            gen_context = self.update_context_text(system_prompt, gen_context)
            edit_cfg_img_context = self.update_context_text(
                system_prompt, edit_cfg_img_context
            )
            output_list.append(system_prompt)

            answer_pattern = r"<answer>(.*?)</answer>"
            depth_estimate_pattern = r"<depth-estimation>(.*?)</depth-estimation>"
            segmentation_pattern = r"<segmentation>(.*?)</segmentation>"

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    edit_cfg_img_context = self.update_context_text(
                        input_term, edit_cfg_img_context
                    )
                    output_list.append(input_term)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(input_term, gen_context)
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                    output_list.append(input_term)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            inter_num = 0
            while True:
                inter_num += 1
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                    top_p=top_p,
                )
                output_list.append(gen_text)
                answer_match = re.search(answer_pattern, gen_text, re.DOTALL)
                if answer_match:
                    return output_list

                if inter_num >= max_inter_num:
                    break

                depth_match = re.search(depth_estimate_pattern, gen_text, re.DOTALL)
                segmentation_match = re.search(
                    segmentation_pattern, gen_text, re.DOTALL
                )

                if depth_match or segmentation_match:
                    if depth_match:
                        entire_depth_prompt = depth_match.group(0)
                        edit_cfg_prompt = gen_text.replace(entire_depth_prompt, "")
                        cfg_text_context = deepcopy(gen_context)
                        cfg_text_context = self.update_context_text(
                            edit_cfg_prompt, cfg_text_context
                        )
                        gen_context = self.update_context_text(gen_text, gen_context)
                        edit_cfg_img_context = self.update_context_text(
                            gen_text, edit_cfg_img_context
                        )
                        img = (
                            self.gen_image(
                                image_shapes,
                                gen_context=gen_context,
                                cfg_text_precontext=cfg_text_context,
                                cfg_img_precontext=edit_cfg_img_context,
                                cfg_text_scale=cfg_text_scale,
                                cfg_img_scale=cfg_img_scale,
                                cfg_interval=cfg_interval,
                                timestep_shift=timestep_shift,
                                num_timesteps=num_timesteps,
                                cfg_renorm_min=cfg_renorm_min,
                                cfg_renorm_type=cfg_renorm_type,
                                out_depth=True,
                            )
                            .cpu()
                            .float()
                        )
                        depth_colored = (
                            colorize_depth_maps(img, 0, 1).squeeze().numpy()
                        )  # [3, H, W], value in (0, 1)
                        depth_colored = (depth_colored * 255).astype(np.uint8)
                        depth_colored_hwc = chw2hwc(depth_colored)
                        img = Image.fromarray(depth_colored_hwc)
                    elif segmentation_match:
                        entire_segmentation_prompt = segmentation_match.group(0)
                        edit_cfg_prompt = gen_text.replace(
                            entire_segmentation_prompt, ""
                        )
                        cfg_text_context = deepcopy(gen_context)
                        cfg_text_context = self.update_context_text(
                            edit_cfg_prompt, cfg_text_context
                        )
                        gen_context = self.update_context_text(gen_text, gen_context)
                        edit_cfg_img_context = self.update_context_text(
                            gen_text, edit_cfg_img_context
                        )
                        img = self.gen_image(
                            image_shapes,
                            gen_context=gen_context,
                            cfg_text_precontext=cfg_text_context,
                            cfg_img_precontext=edit_cfg_img_context,
                            cfg_text_scale=cfg_text_scale,
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval,
                            timestep_shift=timestep_shift,
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                        )

                    output_list.append(pil_img2rgb(img))
                    output_list.append(REFLECTION_PROMPT)
                    img = self.vae_transform.resize_transform(pil_img2rgb(img))
                    gen_context = self.update_context_image(img, gen_context, vae=False)
                    gen_context = self.update_context_text(
                        REFLECTION_PROMPT, gen_context
                    )
                    cfg_text_context = deepcopy(gen_context)
                    edit_cfg_img_context = self.update_context_image(
                        img, edit_cfg_img_context, vae=False
                    )
                    edit_cfg_img_context = self.update_context_text(
                        REFLECTION_PROMPT, edit_cfg_img_context
                    )

        return output_list

    @torch.no_grad()
    def depth_image_generation(
        self,
        input_lists: List[Union[str, Image.Image]],
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        # generate the visualization of the depth map of the image directly
        # the input_list should have the input image and the depth generation prompt: DEPTH_PROMPT

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        edit_cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    edit_cfg_img_context = self.update_context_text(
                        input_term, edit_cfg_img_context
                    )
                    output_list.append(input_term)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(input_term, gen_context)
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                    output_list.append(input_term)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")
            img = (
                self.gen_image(
                    image_shapes,
                    gen_context=gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=edit_cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    out_depth=True,
                )
                .cpu()
                .float()
            )
            depth_colored = (
                colorize_depth_maps(img, 0, 1).squeeze().numpy()
            )  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            img = Image.fromarray(depth_colored_hwc)

        return pil_img2rgb(img)

    @torch.no_grad()
    def segmentation_generation(
        self,
        input_lists: List[Union[str, Image.Image]],
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=30,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        # generate the visualization of the segmentation map of the image directly
        # the input_list should have the input image and the segmentation generation prompt: SEGMENTATION_PROMPT

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        edit_cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    edit_cfg_img_context = self.update_context_text(
                        input_term, edit_cfg_img_context
                    )
                    output_list.append(input_term)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(input_term, gen_context)
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                    output_list.append(input_term)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            img = self.gen_image(
                image_shapes,
                gen_context=gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=edit_cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
            )

        return pil_img2rgb(img)

    @torch.no_grad()
    def image_reason_tool_condition(
        self,
        input_lists: List[Union[str, Image.Image]],
        max_inter_num=3,
        max_think_token_n=2048,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=30,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        top_p=1.0,
        **kwargs,
    ) -> List[Union[str, Image.Image]]:
        # perception enhancement answer function
        # it will first generate the depth map of the image, then generate the segmentation map of the image,
        # then use the depth map and segmentation map to generate the answer without reasoning

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        edit_cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            system_prompt = IMAGE_REASON_SYSTEM_PROMPT
            gen_context = self.update_context_text(system_prompt, gen_context)
            edit_cfg_img_context = self.update_context_text(
                system_prompt, edit_cfg_img_context
            )
            output_list.append(system_prompt)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    edit_cfg_img_context = self.update_context_text(
                        input_term, edit_cfg_img_context
                    )
                    output_list.append(input_term)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(input_term, gen_context)
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                    output_list.append(input_term)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            for it_num in range(3):
                if it_num == 0:
                    cfg_text_context = deepcopy(gen_context)
                    edit_cfg_img_context = self.update_context_text(
                        DEPTH_PROMPT, edit_cfg_img_context
                    )
                    gen_context = self.update_context_text(DEPTH_PROMPT, gen_context)
                    img = (
                        self.gen_image(
                            image_shapes,
                            gen_context=gen_context,
                            cfg_text_precontext=cfg_text_context,
                            cfg_img_precontext=edit_cfg_img_context,
                            cfg_text_scale=cfg_text_scale,
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval,
                            timestep_shift=timestep_shift,
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            out_depth=True,
                        )
                        .cpu()
                        .float()
                    )
                    depth_colored = (
                        colorize_depth_maps(img, 0, 1).squeeze().numpy()
                    )  # [3, H, W], value in (0, 1)
                    depth_colored = (depth_colored * 255).astype(np.uint8)
                    depth_colored_hwc = chw2hwc(depth_colored)
                    img = Image.fromarray(depth_colored_hwc)
                elif it_num == 1:
                    cfg_text_context = deepcopy(gen_context)
                    edit_cfg_img_context = self.update_context_text(
                        SEGMENTATION_PROMPT, edit_cfg_img_context
                    )
                    gen_context = self.update_context_text(
                        SEGMENTATION_PROMPT, gen_context
                    )
                    img = self.gen_image(
                        image_shapes,
                        gen_context=gen_context,
                        cfg_text_precontext=cfg_text_context,
                        cfg_img_precontext=edit_cfg_img_context,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                else:
                    gen_context = self.update_context_text("<answer>", gen_context)
                    gen_text = self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                        top_p=top_p,
                    )
                    output_list.append("<answer>" + gen_text)
                    return output_list

                output_list.append(pil_img2rgb(img))
                img = self.vae_transform.resize_transform(pil_img2rgb(img))
                gen_context = self.update_context_image(img, gen_context, vae=False)
                cfg_text_context = deepcopy(gen_context)
                edit_cfg_img_context = self.update_context_image(
                    img, edit_cfg_img_context, vae=False
                )

    @torch.no_grad()
    def text_reason(
        self,
        input_lists: List[Union[str, Image.Image]],
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        top_p=1.0,
        is_thinking=True,
        **kwargs,
    ) -> List[Union[str, Image.Image]]:
        # reasoning enhancement answer function
        # it will reason with the textual chain-of-thought first, then generate the answer

        output_list = []
        gen_context = self.init_gen_context()

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if is_thinking:
                system_prompt = VLM_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                output_list.append(system_prompt)
            for input_term in input_lists:
                if isinstance(input_term, str):
                    gen_context = self.update_context_text(input_term, gen_context)
                    output_list.append(input_term)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=False
                    )
                    output_list.append(input_term)
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            gen_text = self.gen_text(
                gen_context,
                do_sample=do_sample,
                temperature=text_temperature,
                max_length=max_think_token_n,
                top_p=top_p,
            )
            output_list.append(gen_text)
        return output_list

    def __call__(
        self, image: Optional[Image.Image] = None, text: Optional[str] = None, **kargs
    ) -> Dict[str, Any]:
        output_dict = {"image": None, "text": None}

        if image is None and text is None:
            print("Please provide at least one input: either an image or text.")
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict["image"] = i
            elif isinstance(i, str):
                output_dict["text"] = i
        return output_dict
