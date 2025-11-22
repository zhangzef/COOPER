# build_interleaved_parallel.py  ← 线程并行版本
import argparse, base64, io, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image


def decode_image(b64: str):
    """base64 → bytes，可直接 Image.open(io.BytesIO(...))"""
    # return base64.b64decode(b64)
    return b64


def process_trajectory_step(step: List) -> Tuple[str, List[str]]:
    """
    把单条 trajectory_step 处理成文本（含 <image> 占位）和该条出现的图片字节流列表
    """
    text, *img_objs = step
    images = [decode_image(obj["url"]) for obj in img_objs]

    if images:
        # 使用 'OBSERVATION:' 一分为二
        parts = text.split("OBSERVATION:", 1)
        token_block = "\n".join(["<image>"] * len(images))
        if len(parts) == 2:
            pre, post = parts
            text = f"{pre}OBSERVATION:{token_block}\n{post}"
        else:
            text = f"{text}\n{token_block}\n"

    return text, images


def process_file(path: Path):
    """
    把单个 json 文件转成 instruction_list, image_list
    """
    obj = json.loads(path.read_text(encoding="utf-8"))

    # ------ question -------
    q_text, q_img_obj = obj["question"]
    q_img = decode_image(q_img_obj["url"])

    # ------ trajectory_steps ------
    step_texts, step_imgs = [], []
    for st_i, step in enumerate(obj["trajectory_step"], start=1):
        t, imgs = process_trajectory_step(step)
        step_texts.append(f"**Step {st_i}:** {t}")
        step_imgs.extend(imgs)

    think_block = "<think>" + "\n".join(step_texts) + "</think>"
    # 以 <image> 分割
    instruction_list = [q_text] + [
        s for s in think_block.split("<image>") if s != ""
    ]

    # ------ answer ------
    ans_txt = obj["answer"].replace("ANSWER:", "").strip()
    instruction_list.append(f"<answer>{ans_txt}</answer>")

    # ------ image_list：question 图 + trajectory 图 ------
    image_list = [q_img] + step_imgs

    return json.dumps(instruction_list), json.dumps(image_list)


# ---------- 主逻辑：并行 ----------
def main(json_dir: Path, out_dir: Path, rows_per_file: int, n_threads: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(json_dir.glob("*.json"))
    file_idx, rows_buf = 0, []
    # ❶ 线程池并行读取 / 处理
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {pool.submit(process_file, p): p for p in paths}

        pbar = tqdm(total=len(futures), desc="Processing")
        for fut in as_completed(futures):
            instr, imgs = fut.result()
            rows_buf.append({"instruction_list": instr, "image_list": imgs})
            pbar.update(1)

            # ❷ 每满 rows_per_file 就写一份 parquet
            if len(rows_buf) == rows_per_file:
                df = pd.DataFrame(rows_buf)
                df.to_parquet(
                    out_dir / f"part-{file_idx:04d}.parquet",
                    engine="pyarrow",
                    index=False,
                )
                print(df.head())
                rows_buf.clear()
                file_idx += 1
        pbar.close()

    # 写剩余
    if rows_buf:
        pd.DataFrame(rows_buf).to_parquet(
            out_dir / f"part-{file_idx:04d}.parquet",
            engine="pyarrow",
            index=False,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", default="../../outputs/mathv360k-20250422-reaoning-step-40k", type=Path)
    ap.add_argument("--out_dir", default="./datasets/mathv360k-20250422-reaoning-step-40k-parquet", type=Path)
    ap.add_argument("--rows_per_file", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=os.cpu_count(),
                    help="线程数 (默认=CPU 核心数)")
    args = ap.parse_args()

    main(args.json_dir, args.out_dir, args.rows_per_file, args.threads)