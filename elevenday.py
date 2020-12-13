# 张量的结构操作
# 张量的操作主要包括张量的结构操作和张量的数学运算。
# 张量结构操作诸如：张量创建、索引切片、维度变换、合并分割。
# 张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。
# 本篇我们介绍张量的结构操作。
# 一、创建张量
# 张量创建的许多方法和numpy中创建array的方法很像。
import numpy as np
import torch

a = torch.tensor([1,2,3], dtype=torch.float)
print(a)

b = torch.arange(1,10,step=2)
print(b)

c = torch.linspace(0.0, 2*3.14, 10)
print(c)

d = torch.zeros((3,3))
print(d)

a = torch.ones((3,3), dtype=torch.int)
b = torch.zeros_like(a, dtype=torch.float)

torch.fill_(b, 5)
print(b)


# 均匀随机分布
torch.manual_seed(0)
minval, maxval = 0, 10
a = minval + (maxval-minval)*torch.rand([5])
print(a)


# 正态分布随机
b = torch.normal(mean=torch.zeros(3,3), std=torch.ones(3,3))
print(b)

# 正态分布随机
mean, std = 2, 5
c = std*torch.randn((3,3))+mean
print(c)

# 整数随机排列
d = torch.randperm(20)
print(d)

# 特殊矩阵
I = torch.eye(3,3)  # 单位矩阵
print(I)
t = torch.diag(torch.tensor([1,2,3])) # 对角矩阵
print(t)

# 二、索引切片
# 张量的索引切片方式和numpy几乎是一样的，切片时支持缺省参数和省略号。
# 可以通过索引和切片对部分元素进行修改。
# 此外，对于不规则的切片提取，可以使用torch.index_select, torch.masked_select, torch.take
# 如果要通过修改张量的某些元素得到新的张量，可以使用torch.where, torch.masked_fill, torch.index_fill

# 均匀随机分布
torch.manual_seed(0)
minval, maxval = 0, 10
t = torch.floor(minval + (maxval-minval)*torch.rand([5,5])).int()
print(t)

# 第0行
print(t[0])

# 倒数第一行
print(t[-1])

# 第1行至第3行
print(t[1:4, :])

# 第1行至最后一行，第0列到最后一列每隔两列取一列
print(t[1:4, :4:2])

# 可以使用索引和切片修改部分元素
x = torch.tensor([[1,2],[3,4]], dtype=torch.float32, requires_grad=True)
x.data[1,:] = torch.tensor([0.0, 0.0])
print(x)

a = torch.arange(27).view(3,3,3)
print(a)

# 省略号可以表示多个冒号
print(a[..., 1])

# 以上切片方式相对规则，对于不规则的切片提取，可以使用torch.index_select, torch.take, torch.gather, torch.masked_select.
# 考虑班级成绩册的例子，有4个班级，每个班级10个学生，每个学生7门科目成绩。可以用一个4*10*7的张量来表示。

minval = 0
maxval = 100
scores = torch.floor(minval + (maxval-minval)*torch.rand([4,10,7])).int()
print(scores)

# 抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
torch.index_select(scores, dim=1, index=torch.tensor([0,5,9]))

# 抽取每个班级第0个学生，第5个学生， 第9个学生的第1门课程，第3门课程，第6门课程成绩
q = torch.index_select(torch.index_select(scores, dim=1, index=torch.tensor([0,5,9])),
                       dim=2, index=torch.tensor([1,3,6]))
print(q)

# 抽取第0个班级第0个学生的第0门课程，第2个班级的第4个学生的第1门课程，第3个班级的第9个学生第6门课程成绩
# take将输入看成一堆数组，输出和index同形状。
s = torch.take(scores, torch.tensor([0*10*7+0,2*10*7+4*7+1,3*10*7+9*7+6]))
print(s)

# 抽取分数大于等于80分的分数（布尔索引）
# 结果是1维张量
g = torch.masked_select(scores, scores>=80)
print(g)

# 以上这些方法仅能提取张量的部分元素值，但不能更改张量的部分元素值得到新的张量。
# 如果要通过修改张量的部分元素值得到新的张量，可以使用torch.where, torch.index_fill和torch.masked_fill
# torch.where可以理解为if的张量版本
# torch.index_fill的选取元素逻辑和torch.index_select相同。
# torch.masked_fill的选取元素逻辑和torch.masked_select相同。

# 如果分数大于60分，赋值为1， 否则赋值为0
ifpass = torch.where(scores>60, torch.tensor(1), torch.tensor(0))
print(ifpass)

# 将每个班级第0个学生，第5个学生，第9个学生的全部成绩赋值成满分
torch.index_fill(scores, dim=1, index=torch.tensor([0,5,9]), value=100)
# 等价于scores.index_fill(dim=1, index=torch.tensor([0,5,9]), value=100)

# 将分数小于60分的分数赋值成60分
b = torch.masked_fill(scores, scores<60,60)
# 等价于b=scores.masked_fill(scores<60,60)
print(b)

# 三、维度变换
# 维度变换相关函数主要有torch.reshape（或者调用张量的view方法），torch.squeeze, torch.unsqueezem torch.transpose
# torch.reshape可以改变张量的形状。
# torch.squeeze可以减少维度。
# torch.unsqueeze可以增加维度。
# torch.transpose可以交换维度。

# 张量的view方法有时候会调用失败，可以使用reshape方法。
torch.manual_seed(0)
minval, maxval = 0, 255
a = (minval + (maxval - minval) * torch.rand([1,3,3,2])).int()
print(a.shape)
print(a)

# 改成（3，6）形状的张量
b = a.view([3,6]) # torch.reshape(a, [3,6])
print(b.shape)
print(b)

# 改回成[1,3,3,2]形状的张量
c = torch.reshape(b, [1,3,3,2]) # b.view([1,3,3,2]
print(c)

# 如果张量在某个维度上只有一个元素，利用torch.squeeze可以消除这个维度。
# torch.unsqueeze 的作用和torch.squeeze的作用相反。
a = torch.tensor([[1.0,2.0]])
s = torch.squeeze(a)
print(a)
print(s)
print(a.shape)
print(s.shape)

# 在第0维插入长度为1的一个维度
d = torch.unsqueeze(s, axis=0)
print(s)
print(d)

print(s.shape)
print(d.shape)

# torch.transpose可以交换张量的维度，torch.transpose常用于图片存储格式的变换上。
# 如果是二维的矩阵，通常会调用矩阵的转置方法matrix.t(),等价于torch.transpose(matrix, 0, 1)
minval = 0
maxval = 255
# Batch, Height, Width, Channel
data = torch.floor(minval + (maxval-minval)*torch.rand([100,256, 256, 4])).int()
print(data.shape)

# 转换成Pytorch默认的图片格式Batch，Channel, Height, Width
# 需要交换两次
data_t = torch.transpose(torch.transpose(data, 1, 2), 1, 3)
print(data_t.shape)

matrix = torch.tensor([[1,2,3],[4,5,6]])
print(matrix)
print(matrix.t()) # 等价于torch.transpose(matrix, 0, 1)

# 四、合并分割
# 可以用torch.cat方法和torch.stack方法将多个张量合并，可以用torch.split方法把一个张量分割成多个张量。
# torch.cat和torch.stack有略微的区别，torch.cat是连接，不会增加维度，而torch.stack是堆叠，会增加维度。
a = torch.tensor([[1.0,2.0],[3.0,4.0]])
b = torch.tensor([[5.0,6.0],[7.0,8.0]])
c = torch.tensor([[9.0,10.0],[11.0,12.0]])

abc_cat = torch.cat([a,b,c], dim=0)
print(abc_cat.shape)
print(abc_cat)

abc_stack = torch.stack([a,b,c], axis=0) # torch中dim和axis参数名可以混用
print(abc_stack.shape)
print(abc_stack)

torch.cat([a,b,c], axis=1)

torch.stack([a,b,c], axis=1)

# torch.split是torch.cat的逆运算，可以指定分割数份平均分割，也可以通过指定每份的记录数量进行分割。
print(abc_cat)
a,b,c = torch.split(abc_cat, split_size_or_sections=2, dim=0) # 每份2个进行分割
print(a)
print(b)
print(c)

print(abc_cat)
p,q,r = torch.split(abc_cat, split_size_or_sections=[4,1,1],dim=0) # 每份分别为[4，1，1]
print(p)
print(q)
print(r)

