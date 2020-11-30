import os
import datetime
import importlib
import torchkeras

# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8+"%s"%nowtime)

# mac系统上pytorch和matplotlib在jupyter中同时跑需要更改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 一、准备数据
# 本文的数据取自tushare，获取该数据集的方法参考了以下文章
# 《https://zhuanlan.zhihu.com/p/109556102》

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/convid-9.csv", sep="\t")
df.plot(x="date", y=["confirmed_num", "cured_num", "dead_num"], figsize=(10,6))
plt.xticks(rotation=60)


dfdata = df.set_index("date")
dfdiff = dfdata.diff(period=1).dropna()
dfdiff = dfdiff.reset_index("date")

dfdiff.plot(x="date", y=["confirmed_num", "cured_num", "dead_num"], figsize=(10,6))
plt.xticks(rotation=60)
dfdiff = dfdiff.drop("date", axis=1).astype("float32")

dfdiff.head()

# 下面我们通过继承torch.utils.data.Dataset实现自定义时间序列数据集。
# torch.utils.data.Dataset是一个抽象类，用户想要加载自定义的数据只需要继承这个类，并且覆盖其中的两个方法即可：
# __len__: 实现len(dataset)返回整个数据集的大小。
# __getitem__: 用来获取一些索引的数据，使dataset[i]返回数据集中第i个样本。
# 不覆写这两个方法会直接返回错误

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 用某日前8天窗口数据作为输入预测该日数据
WINDOW_SIZE = 8

class Convid19Dataset(Dataset):

    def __len__(self):
        return len(dfdiff) - WINDOW_SIZE

    def __getitem__(self, i):
        x = dfdiff.loc[i:i+WINDOW_SIZE-1,:]
        feature = torch.tensor(x.values)
        y = dfdiff.loc[i+WINDOW_SIZE,:]
        label = torch.tensor(y.values)
        return (feature, label)

ds_train = Convid19Dataset()

# 数据较小， 可以将全部训练数据放到一个batch中，提升性能
dl_train = DataLoader(ds_train, batch_size=38)

# 二、定义模型
# 使用Pytorch通常有三种方式构建模型：使用nn.Sequential按层顺序构建模型，继承nn.Module基类构建自定义模型，继承nn.Module基类构建
# 模型并辅助应用模型容器进行封装。
# 此处选择第二种方式构建模型。
# 由于接下来使用类形式的训练循环，我们进一步将模型封装成torchkeras中的Model类来获得类似Keras中高阶模型接口的功能。
# Model类实际上继承自nn.Module类。

import torch
from torch import nn
import importlib
import torchkeras

torch.random.seed()

class Block(nn.Module):
    def __init__(self):
        super(Block,self).__init__()

    def forward(self,x,x_input):
        x_out = torch.max((1+x)*x_input[:,-1,:], torch.tensor(0.0))
        return x_out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3层lstm
        self.lstm = nn.LSTM(input_size=3, hidden_size=3, num_layers=5, batch_first=True)
        self.linear = nn.Linear(3,3)
        self.block = Block()

    def forward(self,x_input):
        x = self.lstm(x_input)[0][:,-1,:]
        x = self.linear(x)
        y = self.block(x,x_input)
        return y

net = Net()
model = torchkeras.Module(net)
print(model)

model.summary(input_shape=(8,3), input_dtype=torch.FloatTensor)

# 三、训练模型
# 训练Pytorch通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。
# 有三类典型的训练循环代码风格：脚本形式训练循环，函数形式训练循环，类形式训练循环。
# 此处介绍一种类形式的训练循环。
# 我们仿照Keras定义了一个高阶的模型借口Model，实现fit, validate, predict, summary方法，相当于用户自定义高阶API
# 注： 循环神经网络调试较为困难，需要设置多个不同的学习率多次尝试，以取得较好的效果。
def mspe(y_pred, y_true):
    err_percent = (y_true-y_pred)**2/(torch.max(y_true**2, torch.tensor(1e-7)))
    return torch.mean(err_percent)

model.compile(loss_func=mspe, optimizer=torch.optim.Adagrad(model.parameters(), lr=0.1))

dfhistory = model.fit(100, dl_train, log_step_freq=10)


# 四、评估模型
# 评估模型一般要设置验证集或者测试集，由于此例数较少，我们仅仅可视化损失函数在训练集上的迭代情况。

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.title('Training '+metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric])
    plt.show()

plot_metric(dfhistory, "loss")


# 五、使用模型
# 此处我们使用模型预测疫情结束时间，即 新增确诊病例为0的时间

# 使用dfresult记录现有数据以及此后预测的疫情数据
dfresult = dfdiff[["confirmed_num", "cured_num", "dead_num"]].copy()
dfresult.tail()

# 预测此后200天的新增走势，将其结果添加到dfresult中
for i in range(200):
    arr_input = torch.unsqueeze(torch.from_numpy(dfresult.values[-38:,:]), axis=0)
    arr_predict = model.forward(arr_input)

    dfpredict = pd.DataFrame(torch.floor(arr_predict).data.numpy(),
                             columns=dfresult.columns)
    dfresult = dfresult.append(dfpredict, ignore_index=True)

dfresult.query("confirmed_num==0").head()

dfresult.query("cured_num==0").head()

dfresult.query("head_num==0").head()

# 六、保存模型
# 保存模型参数
print(model.net.state_dict().keys())

torch.save(model.net.state_dict(), "./data/model_parameter.pkl")
net_clone = Net()
net_clone.load_state_dict(torch.load("./data/model_parameter.pkl"))
model_clone = torchkeras.Model(net_clone)
model_clone.compile(loss_func=mspe)

# 评估模型
model_clone.evaluate(dl_train)

