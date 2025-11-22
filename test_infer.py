import os
from PIL import Image
import torch
from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
    init_empty_weights,
)
from inferencer import InterleaveInferencer
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from tqdm import tqdm


## Model Initialization
model_path = "../../models/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
model_param_path = "./results/20250718_225406_interleave_reason_ce_0.25/0003000"
device = "cuda:0"
save_root = "test_infer"

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(
    os.path.join(model_path, "vit_config.json")
)
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
vae_model = vae_model.to(
    device=device,  # 与主模型同卡
    dtype=torch.bfloat16,  # ↙️ 关键：统一成 bfloat16
).eval()
# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
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

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(518, 224, 14)

## Model Loading and Multi GPU Infernece Preparing
max_mem_per_gpu = "80GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.
device_count = torch.cuda.device_count()
device_count = 1
device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(device_count)},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print(device_map)
print(f"num of devices: {device_count}")

same_device_modules = [
    "language_model.model.embed_tokens",
    "time_embedder",
    "latent_pos_embed",
    "vae2llm",
    "llm2vae",
    "connector",
    "vit_pos_embed",
]

if device_count == 1:
    first_device = device_map.get(same_device_modules[0], device)
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = device
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device


model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_param_path, "ema.safetensors"),
    device_map={"": device},
    offload_buffers=False,
    dtype=torch.bfloat16,
)
model = model.eval()
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
    device=device,
)

inference_hyper = dict(
    max_think_token_n=1000,
    max_new_tokns=8192,
    do_sample=False,
    text_temperature=1.0,
    max_inter_num=3,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)

import json
import random

with open("../../outputs/mulberry-20250708-interleaved-jsonl/data.jsonl", "r") as f:
    data_list = f.readlines()


random.seed(42)
data_list = random.sample(data_list, 100)
image_root = "../../outputs/mulberry-20250708-interleaved-jsonl/images"
for item in tqdm(data_list):
    item = json.loads(item)
    data_id = item["data_id"]
    img = Image.open(os.path.join(image_root, item["image"]))
    question = item["question"]
    answer = item["answer"]
    input_lists = [img, question]
    output_list = inferencer.interleave_reason_tool_condition(
        input_lists=input_lists,
        **inference_hyper,
    )
    if len(output_list) <= 2:
        continue
    os.makedirs(f"./{save_root}/{data_id}", exist_ok=True)
    for idx, it in enumerate(output_list):
        if type(it) is not str:
            it.save(f"./{save_root}/{data_id}/{idx}.png")
            output_list[idx] = f"./{save_root}/{data_id}/{idx}.png"
    with open(f"./{save_root}/{data_id}/out.json", "w") as f:
        json.dump({"question": question, "output": output_list, "answer": answer}, f)
    img.save(f"./{save_root}/{data_id}/raw.png")
