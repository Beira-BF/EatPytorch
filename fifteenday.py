# 5-2. 模型层Layers
# 深度学习模型一般由各种模型层组合而成。
# torch.nn中内置了非常丰富的各种模型层。它们都属于nn.Module的子类，具备参数管理功能。
# 例如：
# nn.Linear, nn.Flatten, nn.Dropout, nn.BatchNorm2d
# nn.Conv2d, nn.AvgPool2d, nn.Conv1d, nn.ConvTranspose2d
# nn.Embedding, nn.GRU, nn.LSTM
# nn.Transformer
# 如果这些内置模型层不能够满足需求，我们也可以通过继承nn.Moudle基类构建自定义的模型层。
# 实际上，pytorch不区分模型和模型层，都是通过继承nn.Module进行构建。
# 因此，我们只要继承nn.Module基类并实现forward方法即可自定义模型层。

# 一、内置模型层
import numpy as np
import torch
from torch import nn

# 一些常用的内置模型层简单介绍如下。
# 基础层
# nn.Linear: 全联接层。参数个数=输入层特征数x输出层特征数(weight)+输出层特征数(bias)
# nn.Flatten: 压平层，用于将多维张量样本压成一维张量样本。
# nn.BatchNorm1d: 一维批标准化层。通过线性变换将输入批次缩放平移到稳定的均值和标准差。可以增强模型对输入不同分布的适应性，
# 加快模型训练速度，有轻微正则化效果。一般在激活函数之前使用。可以用afine参数设置该层是否含有可以训练的参数。
# nn.BatchNorm2d: 二维批标准化层。
# nn.BatchNorm3d: 三维批标准化层。
# nn.Dropout: 一维随机丢弃层。一种正则化手段。
# nn.Dropout2d: 二维随机丢弃层。
# nn.Dropout3d: 三维随机丢弃层。
# nn.Threshold: 限幅层。当输入大于或小于阈值范围时，截断之。
# nn.ConstantPad2d: 二维常数填充层。对二维张量样本填充常数扩展长度。
# nn.ReplicationPad1d: 一维复制填充层。对一维张量样本通过复制边缘值填充扩展长度。
# nn.ZeroPad2d: 二维零值填充层。对二维张量样本在边缘填充0值。
# nn.GroupNorm: 组归一化。一种替代批归一化的方法，将通道分成若干组进行归一。不受batch大小限制，据称性能和效果都优于BatchNorm.
# nn.LayerNorm: 层归一化。较少使用。
# nn.InstanceNorm2d: 样本归一化。较少使用。
# 各种归一化技术参考如下知乎文章《FAIR何恺明等人提出组归一化：替代批归一化，不受批量大小限制》https://zhuanlan.zhihu.com/p/34858971

# 卷积网络相关层
# nn.Conv1d: 普通一维卷积，常用于文本。参数个数=输入通道数x卷积核尺寸（如3）x卷积核个数+卷积核尺寸（如3）
# nn.Conv2d: 普通二维卷积，常用于图像。参数个数=输入通道数x卷积核尺寸（如3乘3）x卷积核个数+卷积核尺寸（如3乘3）
# 通过调整dilation参数大于1，可以变成空洞卷积，增大卷积核感受野。通过调整groups参数不为1，可以变成分组卷积。分组卷积中不同分组使用相同
# 的卷积核，显著减少参数数量。当groups参数等于通道数时，相当于tensorflow中的二维深度卷积层tf.keras.layers.DepthwiseConv2D.
# 利用分组卷积和1乘1卷积的组合操作，可以构造相当于Keras中的二维深度可分离卷积层tf.keras.layers.SeparableConv2D.
# nn.Conv3d: 普通三维卷积，常用语视频。参数个数=输入通道数x卷积核尺寸（如3乘3乘3）x卷积核个数+卷积核尺寸（如3乘3乘3）。
# nn.MaxPool2d: 二维最大池化。一种下采样方式。没有需要训练的参数。
# nn.MaxPool3d: 三维最大池化。
# nn.AdaptiveMaxPool2d: 二维自适应最大池化。无论输入图像的尺寸如何变化，输出的图像尺寸是固定的。该函数的实现原理，大概是通过输入
# 图像的尺寸来反向推算池化算子的padding，stride等参数。

