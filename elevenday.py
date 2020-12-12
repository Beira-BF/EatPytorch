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
