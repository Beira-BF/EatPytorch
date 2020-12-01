# 2-1 张量数据结构
# Pytorch的基本数据结构是张量Tensor。张量即多维数组。Pytorch的张量和numpy中的array很类似。
# 本节我们主要介绍张量的数据类型、张量的纬度、张量的尺寸、张量和numpy数组等基本概念。

# 一、张量的数据类型
# 张量的数据类型和numpy.array基本一一对应，但是不支持str类型。
# 包括：
"""
torch.float64(torch.double),
torch.float32(torch.float),
torch.float16,
torch.int64(torch.long),
torch.int32(torch.int),
torch.int16,
torch.int8,
torch.uint8,
torch.bool
"""
# 一般神经网络建模使用的都是torch.float32类型。

import numpy as np
import torch

# 自动推断数据类型

i = torch.tensor(1);print(i,i.dtype)
x = torch.tensor(2.0);print(x,x.dtype)
b = torch.tensor(True);print(b,b.dtype)

# 指定数据类型
i = torch.tensor(1, dtype=torch.int32);print(i,i.dtype)
x = torch.tensor(2.0, dtype=torch.double);print(x, x.dtype)

# 使用特定类型构造函数
i = torch.IntTensor(1);print(i,i.dtype)
x = torch.Tensor(np.array(2.0));print(x,x.dtype) # 等价于torch.FloatTensor
b = torch.BoolTensor(np.array([1,0,2,0])); print(b, b.dtype)

# 不同类型进行转换
i = torch.tensor(1); print(i, i.dtype)
x = i.float(); print(x, x.dtype) # 调用 float方法转换成浮点类型
y = i.type(torch.float); print(y, y.dtype)  # 使用type函数转换成浮点类型
z = i.type_as(x); print(z, z.dtype) # 使用type_as方法转换成某个Tensor相同类型

# 二、张量的纬度
# 不同类型的数据可以用不同纬度（dimension）的张量来表示。
# 标量为0维张量，向量为1维张量，矩阵为2维张量。
# 彩色图像有rgb三个通道，可以表示为3维张量。
# 视频还有时间维，可以表示为4维张量。
# 可以简单地总结为：有几层中括号，就是多少维的张量。

scalar = torch.tensor(True)
print(scalar)
print(scalar.dim()) # 标量，0维张量

vector = torch.tensor([1.0, 2.0, 3.0, 4.0]) # 向量，1维张量
print(vector)
print(vector.dim())

matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # 矩阵，2维张量
print(matrix)
print(matrix.dim())

tensor3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],[[5.0, 6.0], [7.0, 8.0]]]) # 3维张量
print(tensor3)
print(tensor3.dim())

tensor4 = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]],
                       [[[5.0, 5.0], [6.0, 6.0]], [[7.0, 7.0], [8.0, 8.0]]]]) # 4维张量
print(tensor4)
print(tensor4.dim())


# 三、张量的尺寸
# 可以使用shape属性或者size()方法查看张量在每一维的长度。
# 使用view方法改变张量的尺寸。
# 如果view方法改变尺寸失败，可以使用reshape方法。
scalar = torch.tensor(True)
print(scalar.size())
print(scalar.shape)

vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(vector.size())
print(vector.shape)

matrix = torch.tensor([[1.0, 2.0],[3.0, 4.0]])
print(matrix.size())
print(matrix.shape)

# 使用view可以改变张量尺寸

vector = torch.arange(0,12)
print(vector)
print(vector.shape)

matrix34 = vector.view(3,4)
print(matrix34)
print(matrix34.shape)

matrix43 = vector.view(4,-1) # -1表示该位置长度由程序自动推断
print(matrix43)
print(matrix43.shape)

# 不能用以下代码进行view
"""
Traceback (most recent call last):
  File "/baifang/allennlp/EatPytorch/fiveday.py", line 104, in <module>
    matrixhh = vector.view(5, -1)
RuntimeError: shape '[5, -1]' is invalid for input of size 12
"""
# matrixhh = vector.view(5, -1)
# print(matrixhh)
# print(matrixhh.shape)

matrix26 = torch.arange(0,12).view(2,6)
print(matrix26)
print(matrix26.shape)

# 转置操作让张量存储结构扭曲
matrix62 = matrix26.t()
print(matrix62.is_contiguous())

# 直接使用view方法会失败，可以使用reshape方法
# matrix34 = matrix62.view(3,4) # error!
"""
Traceback (most recent call last):
  File "/baifang/allennlp/EatPytorch/fiveday.py", line 124, in <module>
    matrix34 = matrix62.view(3,4) # error!
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
"""
matrix34 = matrix62.reshape(3,4) # 等价于matrix34 = matrix62.contiguous().view(3,4)
print(matrix34)

matrix34 = matrix62.contiguous().view(3,4)
print(matrix34)

# 四、张量和numpy数组
# 可以用numpy方法从Tensor得到numpy数组，也可以用torch.from_numpy从numpy数组得到Tensor。
# 这两种方法关联Tensor和numpy数组是共享数据内存的。
# 如果改变其中一个，另外一个的值也会发生改变。
# 如果有需要，可以用张量的clone方法拷贝张量，中断这种关联。
# 此外，还可以使用item方法从标量张量得到对应的Python数值。
# 使用tolist方法从张量得到对应的Python数值列表。
import numpy as np
import torch

# torch.from_numpy函数从numpy数组得到Tensor
arr = np.zeros(3)
tensor = torch.from_numpy(arr)
print("before add 1:")
print(arr)
print(tensor)
print(tensor.shape)

print("\n after add 1:")
np.add(arr, 1, out=arr) # 给 arr增加1，tensor也随之改变
print(arr)
print(tensor)
print(tensor.size())

# numpy方法从Tensor得到numpy数组
tensor = torch.zeros(3)
arr = tensor.numpy()
print("before add 1:")
print(tensor)
print(arr)

print("\nafter add 1:")

# 使用带下划线的方法表示计算结果会返回给调用 张量
tensor.add_(1) # 给tensor增加1，arr也随之改变
# 或torch.add(tensor, 1, out=tensor)
print(tensor)
print(arr)

tensor1 = 0.0 # 数据类型必须一致，若写成0，则会报错
torch.add(tensor, 1, out=tensor1)
print(tensor1)

# 可以用clone()方法拷贝张量，中断这种关联
tensor = torch.zeros(3)
# 使用clone方法拷贝张量，拷贝后的张量和原始张量内存独立
arr = tensor.clone().numpy() # 也可以使用tensor.data.numpy()
print("before add 1:")
print(tensor)
print(arr)

print("\nafter add 1:")

# 使用带下划线的方法表示计算结果会返回给调用张量
tensor.add_(2) # 给tensor增加2， arr不再随之改变
print(tensor)
print(arr)

arr2 = tensor.data.numpy()
print(arr2)

# item方法和tolist方法可以将张量转换成Python数值和数值列表
scalar = torch.tensor(1.0)
s = scalar.item()
print(s)
print(type(s))

tensor = torch.rand(2,2)
t = tensor.tolist()
print(t)
print(type(t))