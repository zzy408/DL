"""
net.py
颜色分类神经网络定义（ColorNet）
任务：三分类图像分类（blue / red / yellow）
输入尺寸：3 x 64 x 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class mixed_net(nn.Module):
    def __init__(self):
        super().__init__()
        # ==================== 特征提取层（卷积层） ====================
        # 第 1 层卷积 + BatchNorm + ReLU + MaxPool
        # 输入:  [batch_size, 3, 64, 64]   (3通道RGB，64×64像素)
        # 卷积后: [batch_size, 32, 64, 64]  (32个特征图，尺寸不变，padding=2)
        # 池化后: [batch_size, 32, 32, 32]  (尺寸减半)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第 2 层卷积 + BatchNorm + ReLU + MaxPool
        # 输入:  [batch_size, 32, 32, 32]
        # 卷积后: [batch_size, 64, 32, 32]  (通道翻倍，尺寸不变)
        # 池化后: [batch_size, 64, 16, 16]  (尺寸减半)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第 3 层卷积 + BatchNorm + ReLU + MaxPool
        # 输入:  [batch_size, 64, 16, 16]
        # 卷积后: [batch_size, 128, 16, 16]
        # 池化后: [batch_size, 128, 8, 8]
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第 4 层卷积 + BatchNorm + ReLU + MaxPool
        # 输入:  [batch_size, 128, 8, 8]
        # 卷积后: [batch_size, 256, 8, 8]
        # 池化后: [batch_size, 256, 4, 4]
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全局平均池化 Global Average Pooling (GAP)
        # 输入:  [batch_size, 256, 4, 4]
        # 输出:  [batch_size, 256, 1, 1]
        # 作用：将每个通道的空间信息压缩为一个数值，极大减少参数量
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.5)
        # ==================== 分类层（全连接层）====================
        # 第 1 层全连接 + BatchNorm + ReLU
        # 输入:  [batch_size, 256]   (GAP后展平)
        # 输出:  [batch_size, 128]
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.bn_fc = nn.BatchNorm1d(128)
        # 第 2 层全连接（输出层，无激活）
        # 输入:  [batch_size, 128]
        # 输出:  [batch_size, num_classes]  (num_classes=3，对应 blue/red/yellow)
        self.fc2 = nn.Linear(in_features=128, out_features=3)
        # 参数初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """He (Kaiming) 初始化，配合 ReLU 使用，加速收敛。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向传播，每一行都标注了张量的尺寸变化。
        Args:
            x: 输入张量，形状 [batch_size, 3, 64, 64]

        Returns:
            输出 logits，形状 [batch_size, num_classes]
        """
        # ---------- 特征提取 ----------
        # x: [B, 3, 64, 64]
        x = self.conv1(x)           # -> [B, 32, 64, 64]
        x = self.bn1(x)             # -> [B, 32, 64, 64]
        x = F.relu(x)               # -> [B, 32, 64, 64]
        x = self.pool1(x)           # -> [B, 32, 32, 32]

        x = self.conv2(x)           # -> [B, 64, 32, 32]
        x = self.bn2(x)             # -> [B, 64, 32, 32]
        x = F.relu(x)               # -> [B, 64, 32, 32]
        x = self.pool2(x)           # -> [B, 64, 16, 16]

        x = self.conv3(x)           # -> [B, 128, 16, 16]
        x = self.bn3(x)             # -> [B, 128, 16, 16]
        x = F.relu(x)               # -> [B, 128, 16, 16]
        x = self.pool3(x)           # -> [B, 128, 8, 8]

        x = self.conv4(x)           # -> [B, 256, 8, 8]
        x = self.bn4(x)             # -> [B, 256, 8, 8]
        x = F.relu(x)               # -> [B, 256, 8, 8]
        x = self.pool4(x)           # -> [B, 256, 4, 4]

        # 全局平均池化
        x = self.gap(x)             # -> [B, 256, 1, 1]

        # 展平
        x = x.view(x.size(0), -1)   # -> [B, 256]

        x = self.dropout(x)         # -> [B, 256]（训练时随机置零部分神经元）

        # ---------- 分类 ----------
        x = self.fc1(x)             # -> [B, 128]
        x = self.bn_fc(x)           # -> [B, 128]
        x = F.relu(x)               # -> [B, 128]

        x = self.fc2(x)             # -> [B, num_classes]
        return x

