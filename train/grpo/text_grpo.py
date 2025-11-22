# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from math_verify import parse, verify
from train.grpo.bagel_text_grpo_trainer import (
    BagelTextGRPOTrainer,
)
from grpo_data_module import GRPODataset
from trl.trl import (
    GRPOConfig,
    ScriptArguments,
    TrlParser,
)
from transformers import set_seed
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from data.data_utils import add_special_tokens
from modeling.qwen2 import Qwen2Tokenizer
from data.transforms import ImageTransform, DepthImageTransform
from data.output_transfer import OutputTransfer, DataConfig
from safetensors.torch import load_file
from accelerate import init_empty_weights
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from copy import deepcopy

REFLECTION_PROMPT = "\nHere is the result of the depth-estimation/segmentation. Please note that the result of the depth-estimation/segmentation is not always accurate. Please check it carefully. \nNow please continue to think in <think>...</think> and then decide whether to continue to generate the depth-estimation/segmentation in <depth-estimation>...</depth-estimation>/<segmentation>...</segmentation> or give the answer in <answer>...</answer>.\n"


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'"
        },
    )
    jsonl_path: str = field(
        default="",
        metadata={"help": "Path to the JSONL file containing the dataset."},
    )
    image_root: str = field(
        default="",
        metadata={"help": "Root directory containing the images."},
    )


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."},
    )
    model_param_path: str = field(
        default="",
        metadata={"help": "Path of the pretrained BAGEL model."},
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."},
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."},
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."},
    )
    max_latent_size: int = field(
        default=32,
        metadata={
            "help": "Maximum latent grid size (patches per side) for the VAE latent tensor."
        },
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."},
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."},
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={
            "help": "Maximum number of ViT patches along one image side after cropping / resize."
        },
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={
            "help": "Activation function used in the latent-to-text connector MLP."
        },
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={
            "help": "Interpolate positional embeddings when image resolution differs from pre-training."
        },
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={
            "help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."
        },
    )
    vit_rope: bool = field(
        default=False, metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."},
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."},
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={
            "help": "Probability of dropping ViT visual features during training."
        },
    )


@dataclass
class GRPOTrainingArguments(GRPOConfig):
    # --- optimization & scheduler ---
    timestep_shift: float = field(
        default=3.0,
        metadata={
            "help": "Shift applied to diffusion timestep indices (for latent prediction)."
        },
    )
    num_timesteps: int = field(
        default=30,
        metadata={
            "help": "Shift applied to diffusion timestep indices (for latent prediction)."
        },
    )
    save_dir: str = field(
        default="",
        metadata={"help": "Output directory where the trained models will be saved."},
    )
    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."},
    )
    freeze_vit: bool = field(
        default=False, metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={
            "help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."
        },
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."},
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={
            "help": "Duplicate initial MoE experts so each has identical initialisation."
        },
    )
    use_flex: bool = field(
        default=False,
        metadata={
            "help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."
        },
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={
            "help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."
        },
    )
    max_think_token_n: int = field(
        default=4096,
        metadata={
            "help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."
        },
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    answer_list = []
    for i in range(len(completions)):
        answer_list.append(completions[i][-1].strip())

    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(answer_list, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>([\s\S]*?)</answer>", sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>([\s\S]*?)</answer>", content)
                student_answer = (
                    content_match.group(1).strip() if content_match else content
                )
                student_answer = student_answer.strip().lower()
                ground_truth = ground_truth.strip().lower()
                # print(f"ground truth: {ground_truth}, student answer: {student_answer}")
                if len(student_answer) == 0 or len(ground_truth) == 0:
                    reward = 0.0
                elif student_answer.startswith("yes") and ground_truth.startswith(
                    "yes"
                ):
                    reward = 1.0
                elif student_answer.startswith("no") and ground_truth.startswith("no"):
                    reward = 1.0
                elif student_answer[0] == ground_truth[0] and student_answer[0] in [
                    "a",
                    "b",
                    "c",
                    "d",
                    "e",
                ]:
                    reward = 1.0
                elif student_answer == ground_truth:
                    reward = 1.0
                else:
                    reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(
                    f"------------- {current_time} Accuracy reward: {reward} -------------\n"
                )
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # result_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    result_pattern = r"<think>.*?</think>.*"
    seg_pattern = r"<think>.*?</think>\s*<segmentation>.*?</segmentation>"
    depth_pattern = r"<think>.*?</think>\s*<depth-estimation>.*?</depth-estimation>"
    # reason_pattern = r"<think>.*?</think>"
    reward_list = []
    for i in range(len(completions)):
        match_flag = True
        completions[i] = completions[i][3:]
        for j in range(len(completions[i])):
            if type(completions[i][j]) == str and j != len(completions[i]) - 1:
                # reason_match = re.fullmatch(
                #     reason_pattern, completions[i][j].strip(), re.DOTALL
                # )
                if completions[i][j] == REFLECTION_PROMPT:
                    continue
                seg_match = re.fullmatch(
                    seg_pattern, completions[i][j].strip(), re.DOTALL
                )
                depth_match = re.fullmatch(
                    depth_pattern, completions[i][j].strip(), re.DOTALL
                )
                if seg_match or depth_match:
                    match = True
                else:
                    match = False
            elif type(completions[i][j]) == str and j == len(completions[i]) - 1:
                match = re.fullmatch(
                    result_pattern, completions[i][j].strip(), re.DOTALL
                )
            else:
                continue
            if not match:
                match_flag = False
                break
        reward_list.append(0.1 if match_flag else 0.0)
    return reward_list


def accuracy_reward_with_llm(completions, solution, question, **kwargs):
    """
    并行处理多个 prompt 请求，返回布尔结果列表

    Args:
        system_prompt: 系统指令
        base_url: 模型服务地址
        api_key: API 认证密钥
        prompts: 需要处理的 prompt 列表
        model: 模型名称（必需参数，即使自托管服务也需要占位值）
        max_retries: 最大重试次数
        timeout: 单次请求超时时间(秒)
        max_workers: 最大并发线程数

    Returns:
        list[bool]: 每个 prompt 的处理结果（True/False）
    """
    base_url = "http://yq01-inf-hic-k8s-a100-aa24-0020.yq01.baidu.com:8081/v1"
    api_key = "-"
    system_prompt = """
    You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.
    Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here’s how you can accomplish the task:
    INSTRUCTIONS:
    - Focus on the meaningful match between the predicted answer and the correct answer.
    - Consider synonyms or paraphrases as valid matches.
    - Evaluate the correctness of the prediction compared to the answer.
    """
    user_prompt_template = """
    I will give you a question related to an image and the following text as inputs:
    1. **Question Related to the Image**: {Question}
    2. **Ground Truth Answer**: {Ground_Truth}
    3. **Model Predicted Answer**: {Prediction}
    Your task is to evaluate the model’s predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:
    - **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?
    - **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:
    (1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct.
    (2) If the ground truth answer is a definitive answer, strictly compare the model’s prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model’s prediction should be deemed correct.

    **Output Format**:
    Your response should only include True or False indicating the correctness of the prediction: True for correct and False for incorrect. Note that True means the model’s prediction strictly aligns with the ground truth, while False means it does not.
    The format should be flagged: True or False
    """

    def process_single(prompt: str) -> bool:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.0,  # 确保输出确定性
                    n=1,
                )

                # 解析模型响应
                content = response.choices[0].message.content.strip().lower()
                if content == "true":
                    return True
                elif content == "false":
                    return False

            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                print(f"Error: {e}")
                # 可重试的错误类型
                if attempt == max_retries - 1:
                    return False
                time.sleep(2**attempt)  # 指数退避
            except Exception as e:
                print(f"Error: {e}")
                # 其他不可恢复错误
                if attempt == max_retries - 1:
                    return False

        return False

    # 并行处理所有请求
    max_retries = 5
    timeout = 30
    max_workers = 32
    # model = "qwen3-30b-a3b-instruct-2507"
    model = "qwen2.5-vl-72b"
    prompts = []
    continue_list = []
    answer_list = []
    for idx, completion_list in enumerate(completions):
        if type(completion_list[-1]) is not str:
            answer_list.append("-")
            continue_list.append(idx)
            continue
        content = completion_list[-1].strip()
        content_match = re.search(r"<answer>([\s\S]*?)</answer>", content)
        student_answer = content_match.group(1).strip() if content_match else content
        answer_list.append(student_answer)

    for i in range(len(answer_list)):
        prompt = user_prompt_template.format(
            Question=question[i],
            Ground_Truth=solution[i],
            Prediction=answer_list[i],
        )
        prompts.append(prompt)
    results = [False] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single, prompt): idx
            for idx, prompt in enumerate(prompts)
            if idx not in continue_list
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = False
    results_float = [1.0 if r else 0.0 for r in results]

    return results_float


reward_funcs_registry = {
    "accuracy": accuracy_reward_with_llm,
    "format": format_reward,
}


def main(grpo_args, training_args, model_args):
    set_seed(training_args.seed)
    if model_args.model_param_path == "":
        model_args.model_param_path = model_args.model_path

    finetune_from_ema = True
    llm_config = Qwen2Config.from_json_file(
        os.path.join(model_args.model_path, "llm_config.json")
    )
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    language_model = Qwen2ForCausalLM(llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(model_args.model_path, "vit_config.json")
    )
    vit_config.num_hidden_layers = (
        vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
    )
    vit_config.rope = model_args.vit_rope
    vit_model = SiglipVisionModel(vit_config)

    vae_model, vae_config = load_ae(
        local_path=(os.path.join(model_args.model_path, "ae.safetensors"))
    )

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # Setup FSDP and load pretrained model:
    # ema_model = deepcopy(model)
    if finetune_from_ema:
        model_state_dict_path = os.path.join(
            model_args.model_param_path, f"ema.safetensors"
        )
    else:
        model_state_dict_path = os.path.join(
            model_args.model_param_path, f"model.safetensors"
        )
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    model_state_dict.pop("latent_pos_embed.pos_embed")
    model_state_dict.pop("vit_pos_embed.pos_embed")
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(f"model load msg: {msg}")
    del model_state_dict

    vae_transform = DepthImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(518, 224, 14)
    vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
    data_config = DataConfig(
        vae_image_downsample=vae_image_downsample,
        max_latent_size=model_args.max_latent_size,
        vit_patch_size=model_args.vit_patch_size,
        max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
    )
    output_transfer = OutputTransfer(
        tokenizer,
        vae_transform,
        vit_transform,
        data_config,
        training_args.max_num_tokens,
        new_token_ids,
        use_flex=training_args.use_flex,
    )

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in grpo_args.reward_funcs]
    # Load the dataset
    train_set = GRPODataset(
        jsonl_path=grpo_args.jsonl_path, image_root=grpo_args.image_root
    )

    # Initialize the GRPO trainer
    trainer = BagelTextGRPOTrainer(
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
        vae_model=vae_model,
        output_transfer=output_transfer,
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=None,
        # output_record_file=f"./sample_output/{training_args.run_name}.txt",
    )

    # Train and push the model to the Hub
    trainer.train()
    # Save and push to hub
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOTrainingArguments, ModelArguments))
    grpo_args, training_args, model_args = parser.parse_args_and_config()
    main(grpo_args, training_args, model_args)
