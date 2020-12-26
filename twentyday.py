# 6-3, 使用GPU训练模型
# 深度学习的训练过程非常耗时，一个模型训练几个小时是家常便饭，训练几天也是常有的事情，有时候甚至要训练几十天。
# 训练过程的耗时主要来自于两个部分，一部分来自数据准备，另一部分来自参数迭代。
# 当数据准备过程还是模型训练时间的主要瓶颈时，我们可以使用更多进程来准备数据。
# 当参数迭代过程成为训练时间的主要瓶颈时，我们通常的方法是应用GPU来进行加速。
# Pytorch中使用GPU加速模型非常简单，只要将模型和数据移动到GPU上。核心代码只有以下几行。

import torch


# 定义模型
...
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)   # 移动模型到cuda
...

# 训练模型
...
features = features.to(device)  # 移动数据到cuda
labels = labels.to(device)  # 或者   labels= labels.cuda() if torch.cuda.is_available() else labels
...

# 如果要使用多个GPU训练模型，也非常简单。只需要在将模型设置为数据并行风格模型。则模型移动到GPU上之后，会在每一个GPU上拷贝
# 一个副本，并把数据平分到各个GPU上进行训练。核心代码如下。
