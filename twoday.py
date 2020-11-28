import os
import datetime

def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 一、准备数据
# 在Pytorch中构建图片数据管道通常有三种方法
# 第一种使用torchvision中的datasets.ImageFolder来读取图片然后用DataLoader来并行加载。
# 第二种是通过继承torch.utils.data.Dataset实现用户自定义读取逻辑然后用DataLoader来并行加载。
# 第三种方法是读取用户自定义数据集的通用方法，既可以读取图片数据集，也可以读取文本数据集。
# 在此使用第一种方法：

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

transform_train = transforms.Compose([transforms.ToTensor()])
transform_valid = transforms.Compose([transforms.ToTensor()])

ds_train = datasets.ImageFolder("./data/cifar2/train/",
                                transform=transform_train, target_transform=lambda t:torch.tensor([t]).float())
ds_valid = datasets.ImageFolder("./data/cifar2/test/",
                                transform=transform_valid, target_transform=lambda t:torch.tensor([t]).float())
print(ds_train.class_to_idx)

dl_train = DataLoader(ds_train, batch_size=50, shuffle=True, num_workers=3)
dl_valid = DataLoader(ds_valid, batch_size=50, shuffle=True, num_workers=3)

# 查看部分样本
from matplotlib import pyplot as plt

plt.figure(figsize=(8,8))
for i in range(9):
    img,label = ds_train[i]
    img = img.permute(1,2,0)
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label.item())
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# Pytorch的图片默认顺序是Batch, Channel, Width, Height
for x,y in dl_train:
    print(x.shape, y.shape)
    break

# 二、定义模型
# Pytorch通常有三种方式构建模型：使用nn.Sequential按层顺序构建模型，继承nn.Module基类构建自定义模型，继承nn.Module基类构建
# 模型并辅助应用模型容器（nn.Sequential, nn.ModuleList, nn.ModuleDict）进行封装。
# 此处选择通过继承nn.Module基类构建自定义模型。

# 测试AdaptiveMaxPool2d的效果
pool = nn.AdaptiveMaxPool2d((1,1))
t = torch.randn(10,8,32,32)
print(pool(t).shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
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

import pandas as pd
from sklearn.metrics import roc_auc_score

model = net
model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.loss_func = lambda y_pred, y_true: roc_auc_score(y_true.data.numpy(),y_pred.data.numpy())
model.metric_name = "auc"

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

def valid_step(model, features, labels):

    # 预测模式，dropout层不发生作用
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions, labels)
        metric = model.metric_func(predictions, labels)

    return loss.item(), metric.item()

# 测试train_step效果
features, labels = next(iter(dl_train))
train_step(model, features, labels)

def train_model(model, epochs, dl_train, dl_valid, log_step_freq):

    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns = ["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])
    print("Start Training...")
    nowtime = datetime.datetime.now().striftime('%Y-%m-%d %H:%M:%S')
    print("============"*8 + "%s"%nowtime)

    for epoch in range(1, epoch+1):
        # 1.训练循环-----------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(dl_train, 1):

            loss, metric = train_step(model, features, labels)
            # 打印batch级别日志
            loss_sum += loss
            metric_sun += metric
            if step%log_step_freq == 0:
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f")%(step, loss_sum/step, metric_sum/step))

        # 2. 验证循环------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(dl_valid, 1):
            val_loss, val_metric = valid_step(model, features, labels)
            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3. 记录日志------------------------------------------
        info = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch -1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f,"+metric_name+" = %.3f, val_loss = %.3f, "+"val+"+metirc_name+" = %.3f")%info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)

    print("Finished Training...")

    return dfhistory

import matplotlib.pylot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics)+1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(dfhistory, "loss")

plot_metric(dfhistory, "auc")

def predict(model, dl):
    model.eval()
    with torch.no_grad():
        result = torch.cat([model.forward(t[0]) for t in dl])
    return (result.data)

# 预测概率
y_pred_probs = predict(model, dl_valid)
print(y_pred_probs)

# 预测类别
y_pred = torch.where(y_pred_probs>0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
print(y_pred)

# 保存模型
print(model.state_dict().keys())

# 保存模型参数
torch.save(model.state_dict(), "./data/model_parameter_cifar2.pkl")

net_clone = Net()
net_clone.load_state_dict(torch.load("./data/model_parameter_cifar2.pkl"))

predict(net_clone, dl_valid)
