import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from lenet import LeNet


if __name__ == '__main__':
    # 加载数据
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transform)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              transform=transform)
    train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=64, shuffle=True)

    # 创建神经网络、定义损失函数、使用SGD更新网络参数
    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 开始训练
    for epoch in range(30):
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # 前向传播
            outputs = model(inputs)

            # 根据输出计算loss
            loss = criterion(outputs, labels)

            # 梯度清零
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 根据梯度优化网络参数
            optimizer.step()

            if i % 100 == 99:
                print('[Epoch %d, Batch %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss.item()))

    # 开始测试
    correct = 0
    total = 0
    # 测试时前向传播中不记录梯度，节省内存
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            # predicted为每行最大值的索引
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network: %d %%' % (100 * correct / total))
