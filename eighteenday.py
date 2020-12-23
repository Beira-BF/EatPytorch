# 6-1, 构建模型的3种方法
# 可以使用以下3种方法构建模型：
# 1，继承nn.module基类构建自定义模型。
# 2，使用nn.Sequential按层顺序构建模型。
# 3，继承nn.Module基类构建模型并辅助应用模型容器进行封装(nn.Sequential, nn.ModuleList, nn.ModuleDict).
# 其中，第1种方式最为常见，第2种方式最简单，第3种方式最为灵活也较为复杂。
# 推荐使用第1种方式构建模型。

import torch
from torch import nn
from torchkeras import summary

# 一、继承nn.Module基类构建自定义模型
# 以下是继承nn.Module基类构建自定义模型的一个范例。模型中用到的层一般在__init__函数中定义，然后在forward方法中定义模型的正向传播逻辑。

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y

net = Net()
print(net)

summary(net, input_shape=(3,32,32))

# 二、使用nn.Sequtntial按层顺序构建模型
# 使用nn.Sequential按层顺序构建模型无需定义forward方法。仅仅适合于简单的模型。
# 以下是使用nn.Sequential搭建模型的一些等价方法。
# 1，利用add_module方法

net = nn.Sequential()
net.add_module("conv1", nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3))
net.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
net.add_module("conv2", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5))
net.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
net.add_module("dropout", nn.Dropout2d(p=0.1))
net.add_module("adaptive_pool", nn.AdaptiveMaxPool2d((1,1)))
net.add_module("flatten", nn.Flatten())
net.add_module("linear1", nn.Linear(64, 32))
net.add_module("relu", nn.ReLU())
net.add_module("linear2", nn.Linear(32,1))
net.add_module("sigmoid", nn.Sigmoid())

print(net)

# 2, 利用变长参数
# 这种方式构建时不能给每个层指定名称。
net = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout2d(p=0.1),
    nn.AdaptiveMaxPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,1),
    nn.Sigmoid()
)

print(net)

# 3, 利用OrderedDict
from collections import OrderedDict

net = nn.Sequential(OrderedDict(
    [("conv1", nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)),
     ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),
     ("conv2", nn.MaxPool2d(kernel_size=2, stride=2)),
     ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),
     ("dropout", nn.Dropout2d(p=0.1)),
     ("datptive_pool", nn.AdaptiveMaxPool2d((1,1))),
     ("flatten", nn.Flatten()),
     ("linear1", nn.Linear(64,32)),
     ("relu", nn.ReLU()),
     ("linear2", nn.Linear(32,1)),
     ("sigmoid", nn.Sigmoid())
     ])
)
print(net)

# summary(net, input_shape=(3,32,64))

# 三、继承nn.Module基类构建模型并辅助应用模型容器进行封装
# 当模型的结构比较复杂时，我们可以应用模型容器(nn.Sequential, nn.ModuleList, nn.ModuleDict)对模型的部分结构进行封装。
# 这样做会让模型整体更加有层次感，有时候也能减少代码量。
# 注意，在下面的范例中我们每次仅仅使用一种模型容器，但实际上这些模型容器的使用是非常灵活的，可以在一个模型中任意组合
# 任意嵌套使用。
# 1，nn.Sequential作为模型容器

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1,1))
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv(x)
        y = self.dense(x)
        return y

net = Net()
print(net)

# 2, nn.ModuleList作为模型容器
# 注意下面中的ModuleList不能用Python中的列表代替。

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32,1),
            nn.Sigmoid()]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
net = Net()
print(net)

# summary(net, input_shape=(3,32,32))

# 3, nn.ModuleDict作为模型容器
# 注意下面中的ModuleDict不能用Python中的字典代替。

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer_dict = nn.ModuleDict({"conv1": nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
                                         "pool": nn.MaxPool2d(kernel_size=2, stride=2),
                                         "conv2": nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
                                         "dropout": nn.Dropout2d(p=0.1),
                                         "adaptive": nn.AdaptiveMaxPool2d((1,1)),
                                         "flatten": nn.Flatten(),
                                         "linear1": nn.Linear(64,32),
                                         "relu": nn.ReLU(),
                                         "linear2": nn.Linear(32,1),
                                         "sigmoid": nn.Sigmoid()
                                         })
    def forward(self, x):
        layers = ["conv1", "pool", "conv2", "pool", "dropout", "adaptive",
                  "flatten", "linear1", "relu", "linear2", "sigmoid"]
        for layer in layers:
            x = self.layer_dict[layer](x)
        return x

net = Net()
print(net)

summary(net, input_shape=(3,32,32))