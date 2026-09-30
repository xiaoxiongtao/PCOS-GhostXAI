import torch
import torch.nn as nn
import torchvision.models as models
import timm
import time
from fvcore.nn import FlopCountAnalysis
from ghostnet import GhostNet
from ghostnetv2 import GhostNetV2

# ==========================================
# 【关键修改】导入你自行编码好的本地模型
# ==========================================
try:
    # 假设你的文件名叫 Ghostnet.py，里面定义的类名叫 GhostNet
    from ghostnet import GhostNet

    print("成功导入本地自行编码的 GhostNet 模型")
except ImportError as e:
    print(f"警告: 无法导入本地 GhostNet，请检查文件名和类名。错误信息: {e}")


    # 提供一个备用占位，防止代码直接崩溃
    def GhostNet():
        return None

try:
    # 假设你的 GhostNetV2 文件名叫 GhostnetV2.py，类名叫 GhostNetV2
    # 如果你在同一个文件里，可以写 from Ghostnet import GhostNetV2
    from ghostnetv2 import GhostNetV2

    print("成功导入本地自行编码的 GhostNetV2 模型")
except ImportError as e:
    print(f"警告: 无法导入本地 GhostNetV2，请检查文件名和类名。错误信息: {e}")


    def GhostNetV2():
        return None


# ==========================================
# 1. 动态获取模型（已替换本地模型调用）
# ==========================================
def get_model(model_name):
    """
    根据模型名单，自动加载本地模型或官方库模型
    """
    name_lower = model_name.lower().replace("-", "").replace(" ", "")

    # 其余基准标准模型保持不变
    num_classes = 2  # ← 必须和训练时的分类数一样！！！

    if 'resnet' in name_lower:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif 'inceptionresnet' in name_lower or 'inception_res' in name_lower:
        model = timm.create_model('inception_resnet_v2', pretrained=False)
        model.classif = nn.Linear(model.classif.in_features, num_classes)

    elif 'inception' in name_lower:
        model = models.inception_v3(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif 'googlenet' in name_lower:
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # 精准匹配 MobileNet 系列
    elif 'mobilenetv2' in name_lower:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'mobilenetv3small' in name_lower:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif 'mobilenetv3large' in name_lower:
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif 'densenet' in name_lower:
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif 'vit' in name_lower:
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif 'ghostnetv2' in name_lower:
        model = GhostNetV2(num_classes=num_classes)
    elif 'ghostnet' in name_lower:
        model = GhostNet(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model
# ==========================================
# 2. 核心性能分析函数
# ==========================================
def profile_model(model_name, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 针对特殊模型调整输入分辨率
    name_lower = model_name.lower().replace("-", "").replace(" ", "")
    if "inception" in name_lower:
        input_size = (3, 299, 299)
    else:
        input_size = (3, 224, 224)

    print(f"正在测试: {model_name:<20} | 输入大小: {input_size}...", end="", flush=True)

    # 生成模拟输入数据
    dummy_input = torch.randn(batch_size, *input_size).to(device)

    # 清理初始显存缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 实例化模型
    try:
        model = get_model(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f" 失败 (模型加载错误: {e})")
        return None

    # ------------------------------------------
    # A. 计算 Params (M) 和 GFLOPs
    # ------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())

    try:
        flops_analyser = FlopCountAnalysis(model, dummy_input[0:1])
        flops_analyser.unsupported_ops_warnings(False)
        gflops = flops_analyser.total() / 1e9
    except Exception:
        gflops = float('nan')

    # ------------------------------------------
    # B. 测量推理延迟 Latency & 显存 Inf. Mem
    # ------------------------------------------
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    repetitions = 30
    if torch.cuda.is_available():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        latencies = []
        with torch.no_grad():
            for _ in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))
        avg_latency = (sum(latencies) / repetitions) / batch_size
        inf_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(repetitions):
                _ = model(dummy_input)
        avg_latency = ((time.time() - start_time) / repetitions) / batch_size * 1000
        inf_mem = 0

    # ------------------------------------------
    # C. 测量训练显存 Train Mem
    # ------------------------------------------
    train_mem = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        try:
            outputs = model(dummy_input)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            elif isinstance(outputs, dict):
                outputs = list(outputs.values())[0]

            loss = outputs.sum()
            loss.backward()
            optimizer.step()
            train_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        except Exception:
            train_mem = float('nan')

    print(" 完成。")

    del model, dummy_input
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "Model": model_name,
        "Params (M)": round(total_params, 2),
        "GFLOPs": round(gflops, 2) if not torch.isnan(torch.tensor(gflops)) else "N/A",
        "Latency (ms/img)": round(avg_latency, 2),
        "Inf. Mem (MB)": round(inf_mem, 2),
        "Train Mem (MB)": round(train_mem, 2) if not torch.isnan(torch.tensor(train_mem)) else "N/A"
    }


# ==========================================
# 3. 顺序执行并打印最终的对比表格
# ==========================================
if __name__ == "__main__":
    # 11 个模型完整名单
    my_model_list = [
        "GhostNet",  # 对应你本地自行编码的 V1
        "GhostNetV2",  # 对应你本地自行编码的 V2
        "ResNet18",
        "Googlenet",
        "InceptionV3",
        "Inception-ResNetV2",
        "DenseNet121",
        "MobileNetV3 Large",
        "MobileNetV3 Small",
        "MobileNetV2",
        "ViT"
    ]

    results = []
    print("开始顺序测试各个模型...")

    for m_name in my_model_list:
        res = profile_model(m_name, batch_size=32)
        if res is not None:
            results.append(res)

    # 打印最终的汇总表格
    print("\n" + "=" * 85 + "\n 最终实验数据汇总统计表（Batch Size: 32）\n" + "=" * 85)
    print(
        f"{'Model':<20} | {'Params(M)':<10} | {'GFLOPs':<8} | {'Latency(ms/img)':<16} | {'Inf.Mem(MB)':<12} | {'Train.Mem(MB)'}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['Model']:<20} | {r['Params (M)']:<10} | {r['GFLOPs']:<8} | {r['Latency (ms/img)']:<16} | {r['Inf. Mem (MB)']:<12} | {r['Train Mem (MB)']}")
    print("=" * 90)