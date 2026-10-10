# =============================================================================
# 0. 导入依赖库与配置
# =============================================================================
import re
import cv2
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import defaultdict

# 测试同步
# 配置图表样式
plt.style.use('ggplot')

# =============================================================================
# 1. 路径设置 (请根据你的本地路径修改)
# =============================================================================
# 如果你在 Windows 上运行，路径可能类似于 Path('D:/data/PCOS')
dataset_path = Path('/root/autodl-tmp/demo/PCOS')
classes = ['infected', 'noninfected']

if not dataset_path.exists():
    print(f"❌ 错误: 找不到数据集路径 {dataset_path}")
    print("请修改代码中的 dataset_path 为你本地的实际路径。")
    exit()

# =============================================================================
# 2. 数据采集与哈希去重
# =============================================================================
print("正在扫描数据集并计算哈希值...")
data = []
hash_groups = defaultdict(list)

for class_name in classes:
    class_dir = dataset_path / class_name
    for img_path in class_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size

                # 计算 MD5 指纹（去重核心）
                file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
                hash_groups[file_hash].append(img_path)

                data.append({
                    'class': class_name,
                    'resolution': f"{width}x{height}",
                    'format': img_path.suffix[1:].upper(),
                    'size_kb': img_path.stat().st_size / 1024,
                    'path': img_path
                })
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

df = pd.DataFrame(data)

# =============================================================================
# 3. 图像质量（模糊度）分析
# =============================================================================
print("正在评估图像清晰度...")


def calcular_calidad(img_path):
    try:
        # 使用 OpenCV 读取灰度图并计算拉普拉斯方差
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except:
        return None


quality_data = []
for _, row in df.iterrows():
    val = calcular_calidad(row['path'])
    if val is not None:
        quality_data.append(val)
    else:
        quality_data.append(np.nan)

df['blur_score'] = quality_data


# =============================================================================
# 4. 结果展示 (PyCharm 兼容版)
# =============================================================================

def show_results():
    # --- 1. 打印文字摘要 ---
    print("\n" + "=" * 30)
    print("📊 数据集统计摘要")
    print("=" * 30)
    print(f"总图像数: {len(df):,}")
    print(f"类别分布: {df['class'].value_counts().to_dict()}")
    print(f"发现重复组: {sum(len(g) > 1 for g in hash_groups.values()):,} 组")
    print(f"分辨率范围: {df['resolution'].min()} 到 {df['resolution'].max()}")
    print(f"疑似模糊图像: {(df['blur_score'] < 100).sum()} 张 (阈值 < 100)")
    print("=" * 30 + "\n")

    # --- 2. 绘制可视化图表 ---
    fig = plt.figure(figsize=(15, 10))
    fig.canvas.manager.set_window_title('PCOS 数据集分析报告')

    # 子图1: 类别平衡
    ax1 = plt.subplot(2, 2, 1)
    df['class'].value_counts().plot(kind='bar', ax=ax1, color=['#ff9999', '#66b3ff'], rot=0)
    ax1.set_title('Class Balance')

    # 子图2: 分辨率分布 (Top 10)
    ax2 = plt.subplot(2, 2, 2)
    df['resolution'].value_counts().nlargest(10).plot(kind='barh', ax=ax2, color='teal')
    ax2.set_title('Top 10 Resolutions')

    # 子图3: 模糊度分布直方图
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(df['blur_score'].dropna(), bins=50, color='purple', alpha=0.7)
    ax3.axvline(100, color='red', linestyle='--', label='Blur Threshold')
    ax3.set_title('Image Quality (Blur Score)')
    ax3.legend()

    # 子图4: 重复样本展示 (展示第一组重复的图片)
    ax4 = plt.subplot(2, 2, 4)
    dup_groups = [g for g in hash_groups.values() if len(g) > 1]
    if dup_groups:
        sample_dup = dup_groups[0]
        img = plt.imread(sample_dup[0])
        ax4.imshow(img)
        ax4.set_title(f'Sample Duplicate: {sample_dup[0].name}')
    else:
        ax4.text(0.5, 0.5, 'No duplicates found', ha='center')
    ax4.axis('off')

    plt.tight_layout()
    print("正在弹出可视化窗口...")
    plt.show()


if __name__ == "__main__":
    show_results()