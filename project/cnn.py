import os
import torch
# 导入神经网络所需要的函数
import torch.nn as nn
# 导入自动求导机制
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available() # GPU
if use_cuda: # 打印GPU情况
    print(torch.cuda.get_device_name(0), use_cuda)

manualSeed = 2020
torch.manual_seed(manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)


EPOCH = 3
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
def data_set(path):
    # 获得数据集
    # 如果是本地的数据集，可以使用torchvision.datasets.ImageFolder()
    # 如果是下载数据集，root意思是下载好的放在本地的哪个位置
    # transform：PIL类型转换为tensor类型
    train_data = torchvision.datasets.MNIST(root=path, 
                                            train=True,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=DOWNLOAD_MNIST)

    train_loader = Data.DataLoader(dataset=train_data,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True)
            
    test_data = torchvision.datasets.MNIST(root=path,
                                        train=False)

    with torch.no_grad():
        test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)
    test_y = test_data.targets

    return train_loader, test_x, test_y

class CNN(nn.Module):
    def __init__(self):
        # 继承父类
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, 
                                                out_channels=16, 
                                                kernel_size=5,
                                                stride=1,
                                                padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2),  # 省略名字
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平去做全连接
        output = self.out(x)
        return output

def train_test(train_loader, test_x, test_y):
    cnn = CNN() 
    if torch.cuda.device_count() > 1:  # 查看当前电脑的可用的gpu的数量，若gpu数量>1,就多gpu训练
        cnn = torch.nn.DataParallel(cnn)
    if use_cuda:
        cnn = cnn.cuda()
    # Adam算法
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    # 交叉熵
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x,y) in enumerate(train_loader):
            if use_cuda:
                x, y = x.cuda(), y.cuda()

            b_x = Variable(x)
            b_y = Variable(y)
            
            output = cnn(b_x)
            loss = loss_function(output, b_y)
            optimizer.zero_grad()   # 清空上一次
            loss.backward()         # 反向传播
            optimizer.step()

            if step % 100 == 0:
                if use_cuda:
                    test_x, test_y = test_x.cuda(), test_y.cuda()
                c = 0
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                for i in range(test_y.size(0)):
                    if pred_y[i] == test_y[i]:
                        c += 1
                accuracy =  c / test_y.size(0)
                print('Epoch:', epoch, '|Step:', step, '|train loss:%.4f'%loss.data, 
                '|test accuracy:%.4f'%accuracy)

if __name__ ==  '__main__':
    train_loader, test_x, test_y = data_set('./project/mnist/')
    train_test(train_loader, test_x, test_y)