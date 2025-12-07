# COOPER 🧭

<p align="center">
  📄 <a href="https://arxiv.org/pdf/2512.04563">Paper</a> |
  🤖 <a href="https://huggingface.co/Starrrrrry/COOPER">COOPER Model</a> |
  🧠 <a href="https://huggingface.co/Starrrrrry/COOPER-AMG">COOPER-AMG Model</a> |
  📂 <a href="https://huggingface.co/datasets/Starrrrrry/COOPER_Train_Set">COOPER Training Data</a>
</p>

This project provides the **official implementation of COOPER**, a **unified multimodal large language model for visual spatial intelligence** that **cooperatively couples perception and reasoning**. Built on top of the **BAGEL** framework, COOPER endows a single model with **intrinsic perception enhancement** (e.g., **depth estimation** and **semantic segmentation**) and **reasoning enhancement via multimodal chain-of-thought**. We further extend COOPER with **reinforcement learning** and a **cooperative perception–reasoning reward**, enabling the model to **adaptively decide when to “perceive” and when to “reason”** during inference.

<!-- ![motivation](./assets/motivation.png) -->

<p align="center">
  <img src="./assets/motivation.png" width="50%" />
</p>

![model](./assets/model.png)


## 🚀 Key Features

- 🧠 **GRPO Training for BAGEL via TRL**:
    - Fine-tune BAGEL-style multimodal models with RL-style objectives.
    - Optimize perception–reasoning behavior directly from feedback signals.
    - Seamlessly extend from supervised multimodal CoT training to RL-based refinement.

- 📊 **VLMEvalKit Integration for BAGEL**:
    - One-line evaluation on a wide range of multimodal benchmarks.
    - Unified interfaces for dataset loading, inference, and result aggregation.
    - Direct comparison with other VLMs under consistent evaluation protocols.

- 🧩 **[SIBench](https://sibench.github.io/Awesome-Visual-Spatial-Reasoning/) (Single-Image Part) + GPT/Deepseek Answer Extraction**:
    - Fully integrated into **VLMEvalKit** as a first-class evaluation task.
    - Equipped with **GPT/Deepseek-based answer extractors** to:
    - Robustly parse free-form model outputs.
    - Reduce evaluation noise from formatting and phrasing.
    - Provide more accurate and reliable spatial reasoning scores.

---


## 🔥 Quick Start

1️⃣ **Set up environment 🛠️**

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



2️⃣ Download checkpoints and datasets 📥

```bash
cd models
# Download the pretrained BAGEL and its config files.
huggingface-cli download --resume-download --local-dir-use-symlinks False ByteDance-Seed/BAGEL-7B-MoT --local-dir BAGEL-7B-MoT

# Not Necessary
# Download the COOPER-AMG ckpt(training with Auxiliary Modality Generation).
huggingface-cli download --resume-download --local-dir-use-symlinks False Starrrrrry/COOPER-AMG --local-dir COOPER-AMG

# Not Necessary
# Download the COOPER ckpt if you want to inference with COOPER.
huggingface-cli download --resume-download --local-dir-use-symlinks False Starrrrrry/COOPER --local-dir COOPER

# Download the training data(without Hypersim).
# If you want to train the COOPER-AMG, you need to download the Hypersim dataset first(https://github.com/apple/ml-hypersim).
cd ..
huggingface-cli download --resume-download --repo-type dataset Starrrrrry/COOPER_Train_Set --local-dir datasets
cd datasets
# merge the dataset with multiple threads(if you have pigz)(recommended)
cat COOPER_Train_Set.tar.gz.part.* | pigz -d | tar xf -
# OR merge the dataset with single thread(if you don't have pigz)
cat COOPER_Train_Set.tar.gz.part.* | gzip -dc | tar xf -
```



## 🔥 Train & Eval 🧪

### 🏋️ Train

```bash
# Training for Auxiliary Modality Generation from BAGEL.
# Or you can download the COOPER-AMG directly.
sh ./scripts/train_mix.sh

# Training for interleaved reasoning SFT.
sh ./scripts/train_reason_interleave_sft.sh

# Training for interleaved reasoning GRPO.
sh ./scripts/train_reason_interleave_grpo.sh
```



### 📐 Eval

```bash
# You can edit the eval config in /VLMEvalKit/eval_cfg/bagel_with_judge.json.
# Set your openai api key in eval_bagel_with_judge.sh and /VLMEvalKit/.env first.
cd VLMEvalKit
sh eval_bagel_with_judge.sh
```



## 📈 Results

![main_result](./assets/main_result.png)



### 📚 Cases

You can find more cases in the `./assets` folder.

![cases](./assets/cases.png)



![generation_cases](./assets/generation_cases.png)


## ✍️ Citation

```bibtex
@article{zhang2025cooper,
  title={COOPER: A Unified Model for Cooperative Perception and Reasoning in Spatial Intelligence},
  author={Zhang, Zefeng and Hao, Xiangzhao and Tang, Hengzhu and Zhang, Zhenyu and Sheng, Jiawei and Li, Xiaodong and Li, Zhenyang and Gao, Li and Shi, Daiting and Yin, Dawei and others},
  journal={arXiv preprint arXiv:2512.04563},
  year={2025}
}
```