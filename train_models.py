import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy
from ghostnet import GhostNet
from scipy.interpolate import make_interp_spline
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# =========================
# 模型工厂
# =========================
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


# =========================
# 选择模型
# =========================
model_name = "ghostnet"


# -----------------------------
# 参数统计
# -----------------------------
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n========== Model Parameters ==========")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("======================================\n")


# -----------------------------
# 数据清洗
# -----------------------------
train_dir = '/root/autodl-tmp/PCOS_Final_Ready/train'
for root, dirs, files in os.walk(train_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
                    with Image.open(file_path) as img2:
                        img2.load()
            except Exception:
                print(f"Removing corrupted file: {file_path}")
                os.remove(file_path)


# -----------------------------
# 预处理
# -----------------------------
class PCOSPreprocessing(object):
    def __call__(self, img):
        img_np = np.array(img.convert('L'))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_np)
        img_smoothed = cv2.GaussianBlur(img_clahe, (5, 5), 0)
        return Image.fromarray(img_smoothed).convert('RGB')


# -----------------------------
# 输入尺寸
# -----------------------------
if model_name in ["inception", "inception_resnet_v2"]:
    input_size = 299
else:
    input_size = 224


# -----------------------------
# 数据加载
# -----------------------------
data_dir = Path(r'/root/autodl-tmp/PCOS_Final_Ready')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        PCOSPreprocessing(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        PCOSPreprocessing(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(data_dir / x, transform[x]) for x in ['train', 'val']}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False)
}


# -----------------------------
# Loss
# -----------------------------
criterion = nn.CrossEntropyLoss()


# =========================
# 学习率策略
# =========================
if model_name == "vit":
    base_lr = 1e-4
elif model_name == "inception":
    base_lr = 3e-4
elif model_name == "densenet":
    base_lr = 3e-4
else:
    base_lr = 5e-4


# -----------------------------
# 绘图（已加 run_id）
# -----------------------------
def plot_history(history, run_id):
    epochs = np.array(range(1, len(history['train_loss']) + 1))

    if len(epochs) < 4:
        epochs_smooth = epochs
        t_acc, v_acc = history['train_acc'], history['val_acc']
        t_loss, v_loss = history['train_loss'], history['val_loss']
    else:
        epochs_smooth = np.linspace(epochs.min(), epochs.max(), 300)
        t_acc = make_interp_spline(epochs, history['train_acc'])(epochs_smooth)
        v_acc = make_interp_spline(epochs, history['val_acc'])(epochs_smooth)
        t_loss = make_interp_spline(epochs, history['train_loss'])(epochs_smooth)
        v_loss = make_interp_spline(epochs, history['val_loss'])(epochs_smooth)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_smooth, t_acc, label='Training Acc')
    plt.plot(epochs_smooth, v_acc, label='Validation Acc')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_smooth, t_loss, label='Training Loss')
    plt.plot(epochs_smooth, v_loss, label='Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'smoothed_{model_name}_run{run_id}.png', dpi=300)
    plt.show()


# -----------------------------
# 训练
# -----------------------------
def train_model(model, criterion, optimizer, scheduler, run_id, num_epochs=20):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f'best_{model_name}_run{run_id}.pth')

    model.load_state_dict(best_model_wts)

    # 记录结果
    with open("results.txt", "a") as f:
        f.write(f"{model_name}_run{run_id} Best Acc: {best_acc:.4f}\n")

    plot_history(history, run_id)
    return model


# =========================
# 主函数：运行
# =========================
if __name__ == '__main__':
    for run_id in range(1, 2):
        print(f"\n========== Run {run_id} ==========\n")

        model = get_model(model_name, num_classes=2)
        model = model.to(device)
        count_parameters(model)

        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )

        train_model(model, criterion, optimizer, scheduler, run_id, num_epochs=50)