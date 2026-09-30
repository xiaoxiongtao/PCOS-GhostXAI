import torch
import torch.nn as nn
from ghostnet import GhostNet

def print_ghostnet_arch_table(model, input_size=(1, 3, 224, 224)):
    device = next(model.parameters()).device
    x = torch.randn(*input_size).to(device)

    # 严格对应流程图，逐段前向，记录输出shape
    with torch.no_grad():
        # 1. Stem
        x = model.conv_stem(x)
        stem_out = x.clone()
        # 2. Stage1
        x = model.blocks[0](x)
        s1_out = x.clone()
        # 3. Stage2
        x = model.blocks[1](x)
        x = model.blocks[2](x)
        s2_out = x.clone()
        # 4. Stage3
        x = model.blocks[3](x)
        x = model.blocks[4](x)
        s3_out = x.clone()
        # 5. Stage4
        x = model.blocks[5](x)
        s4_out = x.clone()
        # 6. Stage5 + Head（你模型Head在最后一个block里）
        x = model.blocks[6](x)
        head_out = x.clone()
        # 7. GlobalAvgPool
        x = model.global_pool(x)
        pool_out = x.clone()
        # 8. FC
        x = torch.flatten(x, 1)
        x = model.classifier(x)
        fc_out = x.clone()

    # 模块对应（Head直接取最后一个block）
    layer_info = [
        ("Stem", model.conv_stem, stem_out),
        ("Stage1", model.blocks[0], s1_out),
        ("Stage2", nn.Sequential(model.blocks[1], model.blocks[2]), s2_out),
        ("Stage3", nn.Sequential(model.blocks[3], model.blocks[4]), s3_out),
        ("Stage4", model.blocks[5], s4_out),
        ("Stage5+Head", model.blocks[6], head_out),
        ("GlobalAvgPool", model.global_pool, pool_out),
        ("FC", model.classifier, fc_out),
    ]

    print("Table 3. Architecture details of the GhostNet model. Summary of each layer, including type, output shape, and parameter count.\n")
    print(f"{'Layer Name':<20} {'Type':<20} {'Shape of Output':<20} {'Count of Parameters':>20}")
    print("-" * 80)

    total_params = 0
    trainable_params = 0

    for name, module, out_tensor in layer_info:
        layer_type = module.__class__.__name__
        out_shape = str(tuple(out_tensor.shape[1:]))
        layer_param = sum(p.numel() for p in module.parameters())
        total_params += layer_param
        if any(p.requires_grad for p in module.parameters()):
            trainable_params += layer_param

        print(f"{name:<20} {layer_type:<20} {out_shape:<20} {layer_param:>20,}")

    print("-" * 80)
    print(f"{'Totals':<20} {'Total':<20} {'-':<20} {total_params:>20,}")
    print(f"{'':<20} {'Trainable':<20} {'-':<20} {trainable_params:>20,}")
    print(f"{'':<20} {'Non-trainable':<20} {'-':<20} {total_params - trainable_params:>20,}")

if __name__ == "__main__":
    model = GhostNet(num_classes=2)
    print_ghostnet_arch_table(model, input_size=(1, 3, 224, 224))