# 取当前日期＋小时，例如 20250613_14
ts=$(date +"%Y%m%d_%H%M%S")
run_id="${ts}_mix"

model_path_param="./results/20250908_001731_mix/0006000"
num_timesteps=50


# 执行torchrun命令，使用当前子目录作为model_path_param
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=23457 \
  train/pretrain_unified_depth_eval.py \
  --valid_data_path ./datasets/ADE20K_SEG/ade20k_validation_identify.jsonl \
  --valid_image_path ./datasets/ADE20K_SEG/ \
  --valid_dataset_name ade20k \
  --save_to_dir "./results/${run_id}_infer_image/ade20k/" \
  --num_timesteps $num_timesteps \
  --model_path ./models/BAGEL-7B-MoT \
  --model_path_param $model_path_param \
  --num_worker 8


torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=23457 \
  train/pretrain_unified_depth_eval.py \
  --valid_data_path ./datasets/nyu/filename_list_test.jsonl \
  --valid_image_path ./datasets/nyu/ \
  --valid_dataset_name nyu \
  --save_to_dir "./results/${run_id}_infer_image/nyu/" \
  --num_timesteps $num_timesteps \
  --model_path ./models/BAGEL-7B-MoT \
  --model_path_param $model_path_param \
  --num_worker 8
