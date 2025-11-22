# COOPER

This project implements **COOPER**, a unified multimodal large language model for visual spatial intelligence that cooperatively couples perception and reasoning. Initially developed on top of the **BAGEL** framework to endow a single model with intrinsic perception enhancement (depth estimation, semantic segmentation) and reasoning enhancement via multimodal chain-of-thought, we further extend COOPER with reinforcement learning and a cooperative perception–reasoning reward to adaptively decide when to “perceive” and when to “reason” during inference.

![motivation](./assests/motivation.png)

![model](./assests/model.png)



## 🔥 Quick Start

1️⃣ Set up environment

```bash
git clone https://github.com/zhangzef/COOPER.git
cd COOPER
conda create -n cooper python=3.10 -y
conda activate cooper
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
pip intall -e ./transformers-4.54.0
pip install -e ./trl
```



2️⃣ Download checkpoints and datasets

```bash
cd models
# download the pretrained BAGEL and its config files
huggingface-cli download --resume-download --local-dir-use-symlinks False ByteDance-Seed/BAGEL-7B-MoT --local-dir BAGEL-7B-MoT

# download the COOPER-AMG ckpt(training with )
huggingface-cli download --resume-download --local-dir-use-symlinks False Starrrrrry/COOPER-AMG --local-dir COOPER-AMG

# download the COOPER ckpt
huggingface-cli download --resume-download --local-dir-use-symlinks False Starrrrrry/COOPER --local-dir COOPER

# download the training data(without Hypersim)
cd ..
huggingface-cli download --resume-download --repo-type dataset Starrrrrry/COOPER_Train_Set --local-dir datasets
```



## 🔥 Train & Eval

### Train

```bash
# training for Auxiliary Modality Generation from BAGEL
# or you can download the COOPER-AMG directly
sh ./scripts/train_mix.sh

# training for interleaved reasoning SFT
sh ./scripts/train_reason_interleave_sft.sh

# training for interleaved reasoning GRPO
sh ./scripts/train_reason_interleave_grpo.sh
```



### Eval

```bash
# you can edit the eval config in /VLMEvalKit/eval_cfg/bagel_with_judge.json
cd VLMEvalKit
sh eval_bagel_with_judge.sh
```



## Results

![main_result](./assests/main_result.png)



### Cases

![generation_cases](./assests/generation_cases.png)



![seg_comp](./assests/seg_comp.png)



![depth_comp](./assests/depth_comp.png)



![depth_good_0](./assests/depth_good_0.png)



![seg_good_0](./assests/seg_good_0.png)



![depth_bad_0](./assests/depth_bad_0.png)

## ✍️ Citation