import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import timm

from ghostnet import GhostNet


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
# 预处理
# =========================
class PCOSPreprocessing(object):
    def __call__(self, img):
        img_np = np.array(img.convert('L'))
        clahe = cv2.createCLAHE(2.0, (8, 8))
        img_np = clahe.apply(img_np)
        img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
        return Image.fromarray(img_np).convert('RGB')


# =========================
# 自动找最后卷积层
# =========================
def find_last_conv_layer(model):
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d):
            return m
    return None


# =========================
# Grad-CAM
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        output = self.model(x)
        idx = output.argmax(dim=1)

        loss = output[:, idx]
        self.model.zero_grad()
        loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        cam = torch.relu(cam).cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        cam = np.power(cam, 0.6)
        cam = cv2.GaussianBlur(cam, (15, 15), 0)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


# =========================
# 三联图生成
# =========================
def create_triple_figure(img_path, cam, save_name="triple_cam.png", size=224):
    original = cv2.imread(img_path)
    original = cv2.resize(original, (size, size))

    cam_resized = cv2.resize(cam, (size, size))
    heatmap = np.uint8(255 * cam_resized)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

    # 热图叠加
    overlay = cv2.addWeighted(original, 0.45, heatmap_color, 0.55, 0)

    # =========================
    # 病灶边界图
    # =========================
    thresh = np.uint8(cam_resized > 0.6) * 255
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = original.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)

    # =========================
    # 拼接三联图
    # =========================
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Heatmap")
    axs[1].axis("off")

    axs[2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Contour")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()

    print(f"Saved: {save_name}")


# =========================
# 主函数
# =========================
def run_cam(model_name, weight_path, img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name, 2)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    target_layer = find_last_conv_layer(model)
    cam_engine = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        PCOSPreprocessing(),
        transforms.ToTensor(),
        transforms.Normalize([0.485]*3, [0.229]*3)
    ])

    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    cam = cam_engine(input_tensor)

    save_name = f"{model_name}_triple_cam.png"
    create_triple_figure(img_path, cam, save_name)


# =========================
# 运行
# =========================
if __name__ == "__main__":
    model_name = "googlenet"
    weight_path = ("best_googlenet_run1.pth")
    img_path = "/root/autodl-tmp/demo/PCOS_Final_Ready/val/infected/Image_035.jpg"

    run_cam(model_name, weight_path, img_path)