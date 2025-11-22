export NCCL_DEBUG=INFO  # 开启PyTorch分布式调试日志
export NCCL_TIMEOUT=3600000  # 超时时间设为30分钟（根据需要调整）

# 取当前日期＋小时，例如 20250613_14
ts=$(date +"%Y%m%d_%H%M%S")
run_id="${ts}_mix"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=23457 \
  train/pretrain_unified_depth.py \
  --dataset_config_file ./data/configs/dense_predict_mix.yaml \
  --checkpoint_dir "./results/${run_id}" \
  --model_path ./models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from ./models/BAGEL-7B-MoT \
  --finetune_from_hf True \
  --auto_resume False \
  --resume-model-only True \
  --finetune-from-ema True \
  --visual_gen True \
  --visual_und True \
  --freeze_vit True \
  --freeze_llm not \
  --log_every 1 \
  --vit_cond_dropout_prob 0 \
  --vae_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --wandb_name "${run_id}" \
  --wandb_runid "${run_id}" \
  --cpu_offload False \
  --use_flex True \
  --lr 5e-6 \
  --ema 0.995 \
  --ce_weight 1.0 \
  --num_worker 1 \
  --total_steps 6000 \
  --save_every 2000 \
  --expected_num_tokens 10240 \
  --max_num_tokens 10240 \
  --max_num_tokens_per_sample 10240
