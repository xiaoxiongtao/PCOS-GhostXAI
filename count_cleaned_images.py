import os


def count_final_data(base_path):
    splits = ['train', 'val']
    categories = ['infected', 'noninfected']

    print(f"{'Split':<10} | {'Category':<15} | {'Count':<10}")
    print("-" * 40)

    total_count = 0
    stats = {}

    for split in splits:
        stats[split] = {}
        for cat in categories:
            path = os.path.join(base_path, split, cat)
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            stats[split][cat] = count
            total_count += count
            print(f"{split:<10} | {cat:<15} | {count:<10}")

    print("-" * 40)
    print(f"Total cleaned images: {total_count}")
    return stats


# 替换为你清洗后的物理路径
final_stats = count_final_data('/root/autodl-tmp/demo/PCOS_Final_Ready')