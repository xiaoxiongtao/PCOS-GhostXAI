import hashlib
import shutil
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

# =============================================================================
# 1. 路径配置 (根据你的本地环境修改)
# =============================================================================
original_dataset_path = Path(r'/root/autodl-tmp/demo/PCOS')  # 原始下载的数据集
output_ready_path = Path(r'/root/autodl-tmp/demo/PCOS_Final_Ready')  # 清洗后可用于训练的数据集
classes = ['infected', 'noninfected']

# =============================================================================
# 2. 扫描并计算哈希值（查重）
# =============================================================================
print("正在扫描原始数据并进行指纹比对...")
unique_files = {'infected': [], 'noninfected': []}
hash_set = set()

for class_name in classes:
    class_dir = original_dataset_path / class_name
    for img_path in class_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                # 计算文件哈希
                file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
                # 只有指纹唯一的图片才会被保留
                if file_hash not in hash_set:
                    hash_set.add(file_hash)
                    unique_files[class_name].append(img_path)
            except Exception as e:
                print(f"跳过损坏文件 {img_path}: {e}")

# =============================================================================
# 3. 物理分拨：划分 训练集(80%) 和 验证集(20%)
# =============================================================================
print("正在进行数据集物理分拨（训练集 80%, 验证集 20%）...")

for class_name in classes:
    # 划分路径列表
    train_files, val_files = train_test_split(unique_files[class_name], test_size=0.2, random_state=42)

    # 创建对应的物理目录结构
    for split in ['train', 'val']:
        target_dir = output_ready_path / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # 执行拷贝
        files_to_copy = train_files if split == 'train' else val_files
        for f in files_to_copy:
            shutil.copy(f, target_dir / f.name)

print(f"\n✅ 处理完成！")
print(f"清洗后的数据已存入: {output_ready_path.absolute()}")
print(f"结构如下：")
print(f"- {output_ready_path}/train/(infected & noninfected)")
print(f"- {output_ready_path}/val/(infected & noninfected)")