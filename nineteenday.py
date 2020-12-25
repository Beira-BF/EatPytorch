# 6-2, 训练模型的3种方法
# Pytorch通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。
# 有3类典型的训练循环代码风格：脚本形式训练循环，函数形式训练循环，类形式训练循环。
# 下面以minist数据集的分类模型的训练为例，演示这3种训练模型的风格。

# 0，准备数据
import torch
from torch import nn
from torchkeras import summary, Model

import torchvision
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="./data/minist/", train=True, download=True, transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./data/minist/", train=False, download=True, transform=transform)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=4)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=128, shuffle=False, num_workers=4)

print(len(ds_train))
print(len(ds_valid))

# 查看部分样本
from matplotlib import pyplot as plt

plt.figure(figsize=(8,8))
for i in range(9):
    img, label = ds_train[i]
    img = torch.squeeze(img)
    ax = plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# 一、脚本风格
# 脚本风格的训练循环最为常见。

net = nn.Sequential()
net.add_module("conv1", nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3))
net.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
net.add_module("conv2", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5))
net.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
net.add_module("dropout", nn.Dropout2d(p=0.1))
net.add_module("adaptive_pool", nn.AdaptiveMaxPool2d((1,1)))
net.add_module("flatten", nn.Flatten())
net.add_module("linear1", nn.Linear(64,32))
net.add_module("relu", nn.ReLU())
net.add_module("linear2", nn.Linear(32,10))

print(net)

summary(net, input_shape=(1,32,32))

import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_true, y_pred_cls)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)
metric_func = accuracy
metric_name = "accuracy"


epochs = 3
log_step_freq = 100

dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])
print("Start Training...")
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("=========="*8 + "%s"%nowtime)

for epoch in range(1, epochs+1):
    # 1. 训练循环-----------------------------------------------
    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1

    for step, (features, labels) in enumerate(dl_train, 1):

        # 梯度清零
        optimizer.zero_grad()
        # 正向传播求损失
        predictions = net(features)
        loss = loss_func(predictions, labels)
        metric = metric_func(predictions, labels)
        # 反向传播求梯度
        loss.backward()
        optimizer.step()
        # 打印batch级别日志
        loss_sum += loss.item()
        metric_sum += metric.item()
        if step%log_step_freq == 0:
            print(("[step = %d] loss: %.3f, "+metric_name+": %.3f")%
                  (step, loss_sum/step, metric_sum/step))

    # 2，验证循环-------------------------------------------------
    net.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1

    for val_step, (features, labels) in enumerate(dl_valid, 1):
        with torch.no_grad():
            predictions = net(features)
            val_loss = loss_func(predictions, labels)
            val_metric = metric_func(predictions, labels)

        val_loss_sum += val_loss.item()
        val_metric_sum += val_metric.item()

    # 3. 记录日志--------------------------------------------------------
    info = (epoch, loss_sum/step, metric_sum/step,
            val_loss_sum/val_step, val_metric_sum/val_step)
    dfhistory.loc[epoch-1] = info

    # 打印epoch级别日志
    print(("\nEPOCH = %d, loss = %.3f,"+metric_name + \
           " = %.3f, val_loss = %.3f, "+"val_"+metric_name+" = %.3f")
          %info)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"==============="*8 + "%s"%nowtime)

print('Finished Training...')

# 二、函数风格
# 该风格在脚本形式上作了简单的函数封装。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 10)
        ])

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

net = Net()
print(net)

summary(net, input_shape=(1, 32, 32))

import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_true, y_pred_cls)

model = net
model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.loss_func = nn.CrossEntropyLoss()
model.metric_func = accuracy
model.metric_name = "accuracy"


def train_step(model, features, labels):
    # 训练模式，dropout层发生作用
    model.train()
    # 梯度清零
    model.optimizer.zero_grad()
    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)
    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()

    return loss.item(), metric.item()

@torch.no_grad()
def valid_step(model, features, labels):
    # 预测模式，dropout层不发生作用
    model.eval()

    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)

    return loss.item(), metric.item()

# 测试train_step效果
features, labels = next(iter(dl_train))
train_step(model, features, labels)


def train_model(model, epochs, dl_train, dl_valid, log_step_freq):

    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========="*8 + "%s"%nowtime)

    for epoch in range(1, epochs+1):
        # 1. 训练循环---------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(dl_train, 1):
            loss, metric = train_step(model, features, labels)
            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step%log_step_freq == 0:
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                      (step, loss_sum/step, metric_sum/step))

        # 2. 验证循环--------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(dl_valid, 1):
            val_loss, val_metric = valid_step(model, features, labels)

            val_loss_sum += val_loss
            val_metric_sum += val_metric


        info = (epoch, loss_sum/step, metric_sum/step,
                val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f,"+metric_name+\
              " = %.3f, val_loss = %.3f, "+"val_"+metric_name+" = %.3f")
              %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"================"*8+"%s"%nowtime)

    print('Finished Training...')
    return dfhistory

epochs = 3
dfhistory = train_model(model, epochs, dl_train, dl_valid, log_step_freq=100)

# 类风格
# 此处使用torchkeras中定义的模型接口构建模型，并调用compile方法和fit方法训练模型。
# 使用该形式训练模型非常简洁明了，推荐使用该形式。
import torchkeras
class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        ])

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

model = torchkeras.Model(CnnModel())
print(model)

model.summary(input_shape=(1,32,32))

from sklearn.metrics import accuracy_score

def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_true.numpy(), y_pred_cls.numpy())

model.compile(loss_func=nn.CrossEntropyLoss(),
              optimizer=torch.optim.Adam(model.parameters(), lr=0.02),
              metrics_dict={"accuracy":accuracy})

dfhistory = model.fit(3, dl_train=dl_train, dl_val=dl_valid, log_step_freq=100)
