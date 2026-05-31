import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
import os


class MixedNet(nn.Module):
    def __init__(self):
        super(MixedNet, self).__init__()
        # TODO: 需要在这里定义网络层，否则模型没有任何可训练参数

    def forward(self, x):
        '''
        公式： W = (W + 2padding - kernel_w) / stride + 1
        '''
        return x


def evaluate(model, loader, device):
    """评估单个数据集，返回正确数量和总数量"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total


if __name__ == "__main__":
    # 图像转换（变量名不要和导入的 transforms 模块冲突）
    img_transform = transforms.Compose(
        [
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # 超参数设置
    BATCH_SIZE = 1024
    EPOCH = 200

    # 加载数据
    trainset = datasets.ImageFolder(root=r'dataset/train', transform=img_transform)
    testset1 = datasets.ImageFolder(root=r'dataset/test1', transform=img_transform)
    testset2 = datasets.ImageFolder(root=r'dataset/test2', transform=img_transform)

    print(f"训练集图片数量: {len(trainset)}")
    print(f"测试集1图片数量: {len(testset1)}")
    print(f"测试集2图片数量: {len(testset2)}")

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    # 测试集不需要 shuffle
    test_loader1 = DataLoader(testset1, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader2 = DataLoader(testset2, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 创建网络（修正设备选择逻辑）
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"使用设备: {device}")
    net = MixedNet().to(device)

    # 打印网络信息
    summary(net, input_size=(1, 3, 64, 64), device=device)
    print(f'标签对应的ID: {trainset.class_to_idx}')

    # 设置优化器、损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    # 确保保存目录存在
    os.makedirs("pth", exist_ok=True)

    best_correct1 = 0.0

    print("Start")
    for epoch in range(EPOCH):
        net.train()
        train_loss = 0.0

        for batch_id, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 累加该 batch 的总 loss（先乘上 batch 内样本数）
            train_loss += loss.item() * data.size(0)

        # ===== 每个 epoch 结束后统一评估 =====
        if epoch > 50 and (epoch + 1) % 10 == 0:
            net.eval()

            correct1, total1 = evaluate(net, test_loader1, device)
            correct2, total2 = evaluate(net, test_loader2, device)

            c1 = correct1 / total1 * 100
            c2 = correct2 / total2 * 100
            avg_loss = train_loss / len(train_loader.dataset)

            print(
                f"epoch:{epoch + 1}\t"
                f"average_loss:{avg_loss:.5f}\t"
                f"correct1:{c1:.2f}%\t"
                f"correct2:{c2:.2f}%"
            )

            # 保存临时 checkpoint
            temp_path = "pth/model_temp.pth"
            torch.save(net.state_dict(), temp_path)

            # 保存最佳模型（以 testset1 的准确率为准）
            if c1 > best_correct1:
                best_correct1 = c1
                best_path = f"pth/model_best_{best_correct1:.2f}.pth"
                print(f"save {best_path}")
                torch.save(net.state_dict(), best_path)
