import os
import random
import shutil
from pathlib import Path

def main():
    # 设置随机种子，保证可重复性（可选）
    random.seed(42)

    current_dir = Path.cwd()
    output_dir = current_dir / "yolo_classification_dataset"
    train_dir = output_dir / "train"
    test_dir = output_dir / "val"

    # 创建输出目录
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 支持的图片扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # 获取当前路径下所有子目录（排除输出目录自身）
    class_dirs = [d for d in current_dir.iterdir() if d.is_dir() and d.name != output_dir.name]

    if not class_dirs:
        print("错误：当前路径下没有找到任何文件夹。")
        return

    # 存储每个类别的统计信息
    class_stats = []

    print("开始处理...\n")
    for class_dir in class_dirs:
        class_name = class_dir.name

        # 获取该类别下的所有图片文件
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        num_images = len(images)

        # 记录统计信息
        class_stats.append((class_name, num_images))

        if num_images == 0:
            print(f"警告：文件夹 '{class_name}' 中没有图片，已跳过。")
            continue

        # 随机打乱并划分训练/测试集
        random.shuffle(images)
        split_idx = int(num_images * 0.8)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # 创建目标子目录（train/类别 和 test/类别）
        train_class_dir = train_dir / class_name
        test_class_dir = test_dir / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)

        # 复制图片
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        for img in test_images:
            shutil.copy2(img, test_class_dir / img.name)

        print(f"类别 '{class_name}': 共 {num_images} 张 | 训练集 {len(train_images)} | 测试集 {len(test_images)}")

    # 生成文件夹名称列表（数组）
    folder_names = [name for name, _ in class_stats]

    # 输出要求的结果
    print("\n" + "="*50)
    print(f"文件夹名称列表（数组）: {folder_names}")
    print(f"文件夹总个数: {len(folder_names)}")
    print("="*50)

    print(f"\n完成！YOLO 格式数据集已保存至: {output_dir}")

if __name__ == "__main__":
    main()

