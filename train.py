import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from torchinfo import summary
from net import mixed_net


def get_class_weights(dataset):
    """
    根据训练集中各类别样本数量计算逆频率权重，用于缓解类别不平衡问题。
    样本越少的类别，权重越高。
    """
    targets = [label for _, label in dataset.samples]
    class_counts = torch.bincount(torch.tensor(targets), minlength=len(dataset.classes))
    # 权重 = 总样本数 / (类别数 * 该类样本数)，并做平滑处理
    weights = class_counts.float().sum() / (len(dataset.classes) * class_counts.float())
    # 归一化，使最大权重为 1
    weights = weights / weights.max()
    return weights


def evaluate(model, loader, device):
    """评估单个数据集，返回正确数量和总数量。"""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total


def evaluate_per_class(model, loader, device, num_classes=3, class_names=None):
    """
    评估单个数据集，返回：
    - 整体正确率
    - 每一类的正确数量和总数量
    """
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    model.eval()
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, dim=1)
            matches = (predicted == labels)
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += matches[i].item()
                class_total[label] += 1

    total_correct = sum(class_correct)
    total = sum(class_total)
    overall_acc = 100.0 * total_correct / total if total > 0 else 0.0

    per_class_acc = {}
    for i in range(num_classes):
        name = class_names[i] if class_names else f"Class {i}"
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        per_class_acc[name] = (class_correct[i], class_total[i], acc)

    return overall_acc, per_class_acc


def tta_predict(model, images, device, n_augments=5):
    """
    Test Time Augmentation (TTA)：在测试时对同一张图片做多种增强，
    取多次预测的平均值作为最终结果，提升鲁棒性和准确率。
    """
    model.eval()
    # 基础变换后的原图
    base_preds = model(images)
    all_preds = [base_preds]

    # 定义几种测试时的增强
    flips = [
        lambda x: torch.flip(x, dims=[3]),  # 水平翻转
        lambda x: torch.flip(x, dims=[2]),  # 垂直翻转
    ]
    with torch.no_grad():
        for flip in flips:
            aug_imgs = flip(images)
            all_preds.append(model(aug_imgs))
    # 求平均 logits
    avg_preds = torch.stack(all_preds).mean(dim=0)
    return avg_preds


if __name__ == "__main__":
    # ==================== 超参数设置 ====================
    BATCH_SIZE = 64          # 批次大小（适当调小以获得更稳定的梯度）
    EPOCH = 200              # 最大训练轮数
    LR = 0.001               # 初始学习率（AdamW 比 SGD 更稳定，适合较小的 LR）
    WEIGHT_DECAY = 1e-4      # 权重衰减（L2 正则化），防止过拟合
    DROPOUT = 0.5            # Dropout 概率
    LABEL_SMOOTHING = 0.1    # 标签平滑，软化 one-hot 标签，增强泛化
    EARLY_STOPPING_PATIENCE = 30  # 早停耐心值：若 30 个 epoch 无提升则停止
    IMAGE_SIZE = 64

    # ==================== 数据增强策略 ====================
    # 训练时增强：对颜色任务特别加入了 ColorJitter（颜色抖动），
    # 让模型对亮度、对比度、饱和度、色相的变化更鲁棒。
    train_transform = transforms.Compose([
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        transforms.RandomHorizontalFlip(p=0.5),          # 随机水平翻转
        transforms.RandomRotation(degrees=15),           # 随机旋转 ±15 度
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),                                               # 颜色抖动（对颜色分类至关重要）
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # 测试时增强：只做 Resize、ToTensor、Normalize，保持图像原始信息
    test_transform = transforms.Compose([
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # ==================== 加载数据集 ====================
    trainset = datasets.ImageFolder(root=r'dataset/train', transform=train_transform)
    testset1 = datasets.ImageFolder(root=r'dataset/test1', transform=test_transform)
    testset2 = datasets.ImageFolder(root=r'dataset/test2', transform=test_transform)

    print(f"训练集图片数量: {len(trainset)}")
    print(f"测试集1图片数量: {len(testset1)}")
    print(f"测试集2图片数量: {len(testset2)}")
    print(f"类别映射: {trainset.class_to_idx}")

    # 类别名称映射（按 class_to_idx 排序）
    idx_to_class = {v: k for k, v in trainset.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    test_loader1 = DataLoader(testset1, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)
    test_loader2 = DataLoader(testset2, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    # ==================== 设备选择 ====================
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"使用设备: {device}")

    # ==================== 创建网络 ====================
    net = mixed_net(num_classes=3, dropout=DROPOUT).to(device)
    summary(net, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device)

    # ==================== 类别不平衡处理：加权损失 ====================
    class_weights = get_class_weights(trainset).to(device)
    print(f"各类别权重: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    # ==================== 优化器与学习率调度 ====================
    # AdamW：Adam 的改进版，解耦权重衰减与学习率，收敛更稳定
    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # CosineAnnealingLR：余弦退火学习率调度，让学习率按余弦曲线衰减，
    # 有助于跳出局部最优，提升最终收敛精度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)

    # ==================== 训练循环 ====================
    os.makedirs("pth", exist_ok=True)

    best_correct1 = 0.0
    best_epoch = 0
    no_improve_count = 0  # 记录连续无提升的 epoch 数，用于早停

    print("\n========== 开始训练 ==========")
    for epoch in range(EPOCH):
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_id, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计训练指标
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 计算本 epoch 的平均训练损失和准确率
        avg_loss = train_loss / len(trainset)
        train_acc = 100.0 * train_correct / train_total

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # ===== 每 10 个 epoch 评估一次测试集（前 50 轮不评估，先让模型初步收敛）=====
        if epoch >= 50 and (epoch + 1) % 10 == 0:
            net.eval()
            correct1, total1 = evaluate(net, test_loader1, device)
            correct2, total2 = evaluate(net, test_loader2, device)

            c1 = correct1 / total1 * 100
            c2 = correct2 / total2 * 100

            print(
                f"Epoch {epoch + 1:03d}/{EPOCH} | "
                f"Train Loss: {avg_loss:.5f} | Train Acc: {train_acc:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"Test1 Acc: {c1:.2f}% | Test2 Acc: {c2:.2f}%"
            )

            # 保存临时 checkpoint
            temp_path = "pth/model_temp.pth"
            torch.save(net.state_dict(), temp_path)

            # 保存最佳模型（以 testset1 的准确率为准）
            if c1 > best_correct1:
                best_correct1 = c1
                best_epoch = epoch + 1
                best_path = f"pth/model_best_{best_correct1:.2f}.pth"
                print(f"  *** 新的最佳模型！保存到 {best_path}")
                torch.save(net.state_dict(), best_path)
                no_improve_count = 0
            else:
                no_improve_count += 10  # 注意：每 10 个 epoch 评估一次

            # 早停判断
            # if no_improve_count >= EARLY_STOPPING_PATIENCE:
            #     print(f"\n早停触发：连续 {EARLY_STOPPING_PATIENCE} 个 epoch 无提升，停止训练。")
            #     break
        else:
            # 其他 epoch 只打印训练信息
            print(
                f"Epoch {epoch + 1:03d}/{EPOCH} | "
                f"Train Loss: {avg_loss:.5f} | Train Acc: {train_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

    print("\n========== 训练结束 ==========")
    print(f"最佳模型在第 {best_epoch} 个 epoch，Test1 准确率: {best_correct1:.2f}%")

    # ==================== 最终详细评估（打印各类别正确率）====================
    # 加载最佳模型
    best_model_path = f"pth/model_best_{best_correct1:.2f}.pth"
    if os.path.exists(best_model_path):
        net.load_state_dict(torch.load(best_model_path, map_location=device))
        net.to(device)
        net.eval()

        print("\n===== 最佳模型在 Test1 上的详细评估 =====")
        overall1, per_class1 = evaluate_per_class(net, test_loader1, device, num_classes=3, class_names=class_names)
        print(f"整体正确率: {overall1:.2f}%")
        for name, (corr, tot, acc) in per_class1.items():
            print(f"  {name:>10s}: {corr:>3d}/{tot:>3d} = {acc:.2f}%")

        print("\n===== 最佳模型在 Test2 上的详细评估 =====")
        overall2, per_class2 = evaluate_per_class(net, test_loader2, device, num_classes=3, class_names=class_names)
        print(f"整体正确率: {overall2:.2f}%")
        for name, (corr, tot, acc) in per_class2.items():
            print(f"  {name:>10s}: {corr:>3d}/{tot:>3d} = {acc:.2f}%")
