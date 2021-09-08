import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        """
            conv1->max_pool1->conv2->max_pool2->fc1->fc2->fc3
        """
        super(LeNet, self).__init__()
        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是卷积核大小
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 使用2x2的窗口进行最大池化
        self.pool = nn.MaxPool2d(2, 2)

        # 输入是32*32*3的图像
        # 通过conv1输出的结果是28*28*6；通过max_pool1层输出结果是14*14*6
        # 通过conv2输出的结果是10*10*16；通过max_pool2层输出结果是5*5*16
        # 由于max_pool2有16个channel输出，每个feature map大小为5*5，所以全连接层fc1的输入是16*5*5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层fc2输出84张特征图
        self.fc2 = nn.Linear(120, 84)
        # 最终有10类，所以最后一个全连接层fc3输出数量是10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # view函数将张量x变为一维的向量形式，总特征数并不改变，这样全连接层才能处理
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
