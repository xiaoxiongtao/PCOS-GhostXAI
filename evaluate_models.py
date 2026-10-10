import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import cv2
import os
from PIL import Image

from ghostnet import GhostNet
import timm


# ================== 🔥 只需要改这里 ==================
MODEL_NAME = "googlenet"
# 可选：
# "resnet" / "inception" / "ghostnet" / "vit" / "densenet"
# ===================================================


# 1. 预处理类
class PCOSPreprocessing(object):
    def __call__(self, img):
        img_np = np.array(img.convert('L'))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_np)
        img_smoothed = cv2.GaussianBlur(img_clahe, (5, 5), 0)
        return Image.fromarray(img_smoothed).convert('RGB')


# 2. 基础配置
data_dir = Path('/root/autodl-tmp/demo/PCOS_Final_Ready')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自动匹配模型权重
model_path = f'best_{MODEL_NAME}_run1.pth'


# 3. 数据加载（自动适配输入尺寸）
img_size = 299 if MODEL_NAME == "inception" else 224

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    PCOSPreprocessing(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = datasets.ImageFolder(data_dir / 'val', transform=test_transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# ================== 🔥 模型工厂 ==================
def get_model(model_name, num_classes):

    if model_name == "resnet":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "inception":
        model = models.inception_v3(aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "inception_resnet_v2":
        import timm
        model = timm.create_model('inception_resnet_v2', pretrained=False)
        model.classif = nn.Linear(model.classif.in_features, num_classes)

    elif model_name == "googlenet":
        model = models.googlenet(aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "ghostnet":
        model = GhostNet(num_classes=num_classes)

    elif model_name == "vit":
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif model_name == "densenet":
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")


    return model
# =================================================


# --- 绘图函数区 ---

def plot_confusion_matrix(y_true, y_pred, classes):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.savefig(f'1_{MODEL_NAME}_Confusion_Matrix.png', dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.savefig(f'3_{MODEL_NAME}_ROC_Curve.png', dpi=300)
    plt.close()


def plot_pr_curve(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR (AUC = {pr_auc:.4f})')
    plt.fill_between(recall, precision, alpha=0.2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.savefig(f'4_{MODEL_NAME}_PR_Curve.png', dpi=300)
    plt.close()


# ================== 主评估流程 ==================
def evaluate():

    model = get_model(MODEL_NAME, num_classes=2)

    if not os.path.exists(model_path):
        print(f"找不到模型文件 {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    all_preds, all_labels, all_probs = [], [], []

    print(f"正在执行 {MODEL_NAME} 模型推理...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])

    classes = test_data.classes

    print("正在生成所有科研图表...")
    plot_confusion_matrix(all_labels, all_preds, classes)
    plot_roc_curve(all_labels, all_probs)
    plot_pr_curve(all_labels, all_probs)

    print(f"\n✅ {MODEL_NAME} 评估完成！")


if __name__ == '__main__':
    evaluate()