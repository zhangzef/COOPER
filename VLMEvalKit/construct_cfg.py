import os
import json
import argparse
from pathlib import Path


def generate_model_config(
    address_list,
    output_path,
    vq_ckpt_path="../models/Chameleon_Tokenizer/vqgan.ckpt",
    vq_cfg_path="../models/Chameleon_Tokenizer/vqgan.yaml",
    max_new_tokens=8192,
    is_cot=True,
    include_root=False,
):
    """
    生成模型配置JSON文件

    参数：
    address_list: 地址列表（目录路径列表）
    output_path: 输出JSON文件路径
    vq_ckpt_path: VQ检查点路径（默认值）
    vq_cfg_path: VQ配置路径（默认值）
    max_new_tokens: 最大新token数（默认8192）
    is_cot: 是否为思维链模型（默认True）
    include_root: 是否包含根目录自身（默认False）
    """

    model_config = {}
    base_dir = Path(output_path).parent.resolve()  # 获取输出目录基准路径

    def default_key_generator(parent_dir, subdir):
        """默认key生成逻辑"""
        parent_name = Path(parent_dir).name
        subdir_name = subdir.name if isinstance(subdir, Path) else subdir

        # 提取父目录特征（示例逻辑，可根据需要修改）
        features = parent_name.split("-")
        base = features[0] if features else "Unknown"
        suffix = features[-1] if len(features) > 1 else ""

        # 提取子目录编号
        num = "".join(filter(str.isdigit, subdir_name)) or "0"

        return f"{base}-Cot-{suffix}-{num}"

    # 遍历所有地址
    for root in address_list:
        root_path = Path(root).resolve()

        # 验证目录是否存在
        if not root_path.exists():
            print(f"警告：跳过不存在的目录 {root_path}")
            continue

        # 遍历目录树
        for dirpath, dirnames, _ in os.walk(root_path):
            current_dir = Path(dirpath)

            # 处理当前目录或其子目录
            targets = [current_dir] if include_root else []
            targets.extend(current_dir.iterdir())

            for target in targets:
                if not target.is_dir():
                    continue

                # 生成配置项
                parent_dir = current_dir
                subdir = target.name

                # 使用自定义生成器或默认逻辑
                key = default_key_generator(parent_dir, subdir)

                # 构建相对路径
                rel_path = os.path.relpath(target, base_dir).replace("\\", "/")

                # 构建配置条目
                model_config[key] = {
                    "class": "Liquid",
                    "model_path": rel_path,
                    "vq_ckpt_path": vq_ckpt_path,
                    "vq_cfg_path": vq_cfg_path,
                    "max_new_tokens": max_new_tokens,
                    "is_cot": is_cot,
                }

    # 构建完整配置（data部分留空待扩展）
    full_config = {
        "model": model_config,
        "data": {
            "MathVista_MINI": {
                "class": "MathVista",
                "dataset": "MathVista_MINI"
            },
            # "MathVision_MINI": {
            #     "class": "MathVision",
            #     "dataset": "MathVision_MINI"
            # },
            "MMBench_DEV_EN": {
                "class": "ImageMCQDataset",
                "dataset": "MMBench_DEV_EN"
            },
            "MMMU_DEV_VAL": {
                "class": "MMMUDataset",
                "dataset": "MMMU_DEV_VAL"
            },
            # "MathVerse_MINI": {
            #     "class": "MathVerse",
            #     "dataset": "MathVerse_MINI"
            # },
            "MM-Math": {
                "class": "MMMath",
                "dataset": "MM-Math"
            },
            # "MMStar_MINI":{
            #     "class": "ImageMCQDataset",
            #     "dataset": "MMStar_MINI"
            # },
            # "MMVet": {
            #     "class": "MMVet",
            #     "dataset": "MMVet"
            # },
            # "HallusionBench": {
            #     "class": "ImageYORNDataset",
            #     "dataset": "HallusionBench"
            # }
        },
    }

    # 写入JSON文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成模型配置JSON文件")
    parser.add_argument(
        "--addresses",
        nargs="+",
        required=True,
        help="要处理的目录地址列表（多个地址用空格分隔）",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出JSON文件路径",
    )
    parser.add_argument(
        "--vq_ckpt",
        default="../models/Chameleon_Tokenizer/vqgan.ckpt",
        help="VQ检查点路径（默认: %(default)s）",
    )
    parser.add_argument(
        "--vq_cfg",
        default="../models/Chameleon_Tokenizer/vqgan.yaml",
        help="VQ配置路径（默认: %(default)s）",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="最大新token数（默认: %(default)s）",
    )
    parser.add_argument(
        "--is_cot",
        action="store_true",
        help="标记为思维链模型（默认: True）",
    )
    parser.add_argument(
        "--no_cot",
        action="store_false",
        dest="is_cot",
        help="标记为非思维链模型",
    )

    args = parser.parse_args()

    # 参数验证
    if not args.addresses:
        parser.error("必须至少指定一个目录地址")

    # 转换路径为绝对路径
    address_list = [Path(addr).resolve() for addr in args.addresses]
    output_path = Path(args.output).resolve()

    # 执行生成
    generate_model_config(
        address_list=address_list,
        output_path=output_path,
        vq_ckpt_path=args.vq_ckpt,
        vq_cfg_path=args.vq_cfg,
        max_new_tokens=args.max_tokens,
        is_cot=args.is_cot,
    )

    print(f"配置文件已生成：{output_path}")
