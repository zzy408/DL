import torch
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from net import mixed_net


def test_model(model_path, test_loader, device, class_names=None, use_tta=False):
    """
    使用训练好的模型对测试集进行验证，打印整体正确率和每一类的正确率。

    Args:
        model_path: 模型权重文件路径 (.pth)
        test_loader: 测试数据加载器
        device: 计算设备 (cpu / cuda / mps)
        class_names: 类别名称列表，如 ['blue', 'red', 'yellow']
        use_tta: 是否启用 Test Time Augmentation(测试时增强)
    """
    # 加载模型权重
    net = mixed_net()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    num_classes = len(class_names) if class_names else 3
    class_correct = [0] * 3
    class_total = [0] * 3

    print(f"\n模型路径: {model_path}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"TTA (测试时增强): {'开启' if use_tta else '关闭'}")

    # 开始测试
    with torch.no_grad():
        for datas, labels in test_loader:
            datas, labels = datas.to(device), labels.to(device)

            if use_tta:
                # Test Time Augmentation：对 batch 内图片做多种翻转后取平均
                outputs = tta_predict(net, datas)
            else:
                outputs = net(datas)

            _, predicted = torch.max(outputs.data, dim=1)

            matches = (predicted == labels)
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += matches[i].item()
                class_total[label] += 1

    # 计算并打印整体正确率
    total_correct = sum(class_correct)
    total = sum(class_total)
    overall_accuracy = 100.0 * total_correct / total if total > 0 else 0.0
    print(f"\n>>> 整体正确率 (Overall Accuracy): {overall_accuracy:.2f}%")

    # 打印每一类的正确率
    print(">>> 每一类的正确率 (Per-Class Accuracy):")
    for i in range(num_classes):
        name = class_names[i] if class_names else f"Class {i}"
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        print(f"    {name:>10s}: {class_correct[i]:>3d}/{class_total[i]:>3d} = {acc:.2f}%")

    return overall_accuracy


def tta_predict(model, images):
    """
    Test Time Augmentation (TTA)：
    对输入图片分别做：原图、水平翻转、垂直翻转，
    取三次预测 logits 的均值作为最终结果。
    """
    # 原图预测
    preds = [model(images)]

    # 水平翻转预测
    preds.append(model(torch.flip(images, dims=[3])))

    # 垂直翻转预测
    preds.append(model(torch.flip(images, dims=[2])))

    # 取平均
    return torch.stack(preds).mean(dim=0)


if __name__ == "__main__":
    # ==================== 设备选择 ====================
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"使用设备: {device}")

    # ==================== 图像预处理 ====================
    IMAGE_SIZE = 64
    test_transform = transforms.Compose([
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # ==================== 加载数据 ====================
    BATCH_SIZE = 64
    testset1 = datasets.ImageFolder(root=r'dataset/test1', transform=test_transform)
    testset2 = datasets.ImageFolder(root=r'dataset/test2', transform=test_transform)

    test_loader1 = DataLoader(testset1, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader2 = DataLoader(testset2, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 类别名称映射
    idx_to_class = {v: k for k, v in testset1.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"类别映射: {testset1.class_to_idx}")

    # ==================== 模型路径 ====================
    # 默认使用最佳模型，若不存在则使用临时模型
    model_path = r"pth/model_best_100.00.pth"
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        # CPU 环境下可能需要调整路径，这里保持不变
        pass

    # 如果最佳模型不存在，尝试查找其他 .pth 文件
    import os
    if not os.path.exists(model_path):
        pth_dir = "pth"
        if os.path.exists(pth_dir):
            pth_files = sorted([f for f in os.listdir(pth_dir) if f.endswith('.pth')], reverse=True)
            if pth_files:
                model_path = os.path.join(pth_dir, pth_files[0])
                print(f"未找到默认模型，自动选择: {model_path}")
            else:
                print("错误：未找到任何 .pth 模型文件，请先运行 train.py 训练模型！")
                exit(1)
        else:
            print("错误：pth 目录不存在，请先运行 train.py 训练模型！")
            exit(1)

    # ==================== 测试 Test1 ====================
    print("\n" + "=" * 50)
    print("正在测试数据集: test1")
    print("=" * 50)
    test_model(model_path, test_loader1, device, class_names=class_names, use_tta=True)

    # ==================== 测试 Test2 ====================
    print("\n" + "=" * 50)
    print("正在测试数据集: test2")
    print("=" * 50)
    test_model(model_path, test_loader2, device, class_names=class_names, use_tta=True)