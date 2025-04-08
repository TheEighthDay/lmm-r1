import os
import argparse
from huggingface_hub import HfApi, create_repo, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
from tqdm import tqdm
import shutil
import json
from datetime import datetime

def push_to_huggingface(
    model_path,
    repo_name,
    token,
    organization=None,
    private=False,
    commit_message=None
):
    """
    将本地模型推送到HuggingFace
    
    Args:
        model_path: 本地模型路径
        repo_name: 仓库名称
        token: HuggingFace token
        organization: 组织名称（可选）
        private: 是否设为私有仓库
        commit_message: 提交信息
    """
    # 初始化API
    api = HfApi(token=token)
    
    # 构建完整的仓库名称
    if organization:
        full_repo_name = f"{organization}/{repo_name}"
    else:
        full_repo_name = repo_name
    
    # 创建仓库
    try:
        create_repo(
            repo_id=full_repo_name,
            token=token,
            private=private,
            exist_ok=True
        )
    except Exception as e:
        print(f"创建仓库时出错: {str(e)}")
        return False
    
    # 准备模型文件
    print("准备模型文件...")
    temp_dir = "temp_model"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # 复制模型文件
    print("复制模型文件...")
    for item in os.listdir(model_path):
        s = os.path.join(model_path, item)
        d = os.path.join(temp_dir, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)
        elif os.path.isdir(s):
            shutil.copytree(s, d)
    
    # 添加模型卡片
    print("创建模型卡片...")
    model_card = f"""---
language: zh
license: apache-2.0
tags:
- vision-language
- location-recognition
---

# {repo_name}

这是一个基于Qwen2.5-VL-7B的位置识别模型。

## 模型描述

该模型用于识别图片中的地理位置信息，包括国家（country）和一级行政区（administrative_area_level_1）。

## 使用方法

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("{full_repo_name}")
processor = AutoProcessor.from_pretrained("{full_repo_name}")
```

## 评估结果

- 准确率: 待补充
- 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(os.path.join(temp_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)
    
    # 上传文件
    print("上传文件到HuggingFace...")
    try:
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=full_repo_name,
            token=token,
            commit_message=commit_message or f"Upload model files - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"模型已成功上传到: {full_repo_name}")
        return True
    except Exception as e:
        print(f"上传文件时出错: {str(e)}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description='将本地模型推送到HuggingFace')
    parser.add_argument('--model_path', type=str, required=True,
                        help='本地模型路径')
    parser.add_argument('--repo_name', type=str, required=True,
                        help='仓库名称')
    parser.add_argument('--token', type=str, required=True,
                        help='HuggingFace token')
    parser.add_argument('--organization', type=str, default=None,
                        help='组织名称（可选）')
    parser.add_argument('--private', action='store_true',
                        help='是否设为私有仓库')
    parser.add_argument('--commit_message', type=str, default=None,
                        help='提交信息')
    
    args = parser.parse_args()
    
    print("\n开始推送模型到HuggingFace...")
    print(f"模型路径: {args.model_path}")
    print(f"仓库名称: {args.repo_name}")
    if args.organization:
        print(f"组织名称: {args.organization}")
    print(f"私有仓库: {'是' if args.private else '否'}")
    
    success = push_to_huggingface(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token,
        organization=args.organization,
        private=args.private,
        commit_message=args.commit_message
    )
    
    if success:
        print("\n模型推送成功！")
    else:
        print("\n模型推送失败！")

if __name__ == "__main__":
    main()

# 使用示例:
# python push_to_huggingface.py \
#     --model_path /data/phd/tiankaibin/lmm-r1/finetune/lora_merged_model \
#     --repo_name SeekWorld_SFT \
#     --token xx\
#     --organization SeekWorld \
#     --private \
#     --commit_message "Initial model upload"

#     --model_path /data/phd/tiankaibin/experiments_multi_system/checkpoints/lmm-r1-multi-system/ckpt/global_step150_hf \
#     /data/phd/tiankaibin/lmm-r1/experiments_multi/checkpoints/lmm-r1-multi/ckpt/global_step260_hf


#     /data/phd/tiankaibin/lmm-r1/finetune/lora_merged_model



# python push_to_huggingface.py \
#     --model_path /data/phd/tiankaibin/experiments_seekworld_system_length_kl0_2/checkpoints/lmm-r1-seekworld-system-length-kl0_2 \
#     --repo_name SeekWorld_RL \
#     --token xx \
#     --organization SeekWorld \
#     --private \
#     --commit_message "Initial model upload"