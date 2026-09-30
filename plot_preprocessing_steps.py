import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def generate_sci_comparison(image_path, save_name="figure1_preprocessing.png"):
    # 1. 加载原图 (转灰度)
    # 如果路径包含中文，请确保使用支持中文的读取方式
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        print("错误：无法读取图片，请检查路径。")
        return

    # 2. CLAHE 处理
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(raw_img)

    # 3. 高斯平滑处理
    img_smoothed = cv2.GaussianBlur(img_clahe, (5, 5), 0)

    # 4. 创建 SCI 标准对比图
    # 设置全屏宽度，保证三张图横向排列
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    display_list = [
        (raw_img, "(a) Original Ultrasound Image"),
        (img_clahe, "(b) Enhanced by CLAHE"),
        (img_smoothed, "(c) After Gaussian Smoothing")
    ]

    for ax, (img, title) in zip(axes, display_list):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.axis('off')  # 隐藏坐标轴

    # 自动调整间距，防止标题重叠
    plt.tight_layout()

    # 保存为 300 DPI 的高清晰度图片，适合投稿
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"✅ SCI对比图已生成并保存为: {save_name}")
    plt.show()
if __name__ == '__main__':
    generate_sci_comparison('/root/autodl-tmp/demo/PCOS_Final_Ready/val/infected/Image_035.jpg')

# 示例调用 (请将 'test_pcos.jpg' 替换为你清洗后的任意一张图片路径)
