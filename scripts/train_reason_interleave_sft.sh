ts=$(date +"%Y%m%d_%H%M%S")
run_id="reason_interleave_sft_${ts}"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=23456 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/reason_interleave_dataset.yaml \
  --checkpoint_dir "./results/${run_id}" \
  --model_path ./models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from ./results/20250908_001731_mix/0006000 \
  --finetune_from_hf True \
  --auto_resume False \
  --resume-model-only True \
  --finetune-from-ema True \
  --visual_gen True \
  --visual_und True \
  --freeze_vit True \
  --freeze_vae True \
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
  --total_steps 600 \
  --save_every 300 \
  --expected_num_tokens 10240 \
  --max_num_tokens 10240 \
  --max_num_tokens_per_sample 10240