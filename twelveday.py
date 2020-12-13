# 4-2 张量的数学运算
# 张量的操作主要包括张量的结构操作和张量的数学运算。
# 张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。
# 张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。
# 本篇我们介绍张量的数学运算。
# 本篇文章的部分内容参考如下博客：https://blog.csdn.net/duan_zhihua/article/details/82526505

# 一、标量运算
# 张量的数学运算符可以分为标量运算符、向量运算符、以及矩阵运算符。
# 加减乘除乘方，以及三角函数，指数，对数等常见函数，逻辑比较运算符等都是标量运算符。
# 标量运算符的特点是对张量实施逐元素运算。
# 有些标量运算符对常用的数学运算符进行了重载。并且支持类似numpy的广播特性。
import torch
import numpy as np

a = torch.tensor([[1.0, 2], [-3, 4.0]])
b = torch.tensor([[5.0, 6], [7.0, 8.0]])
print(a + b)   # 运算符重载

print(a-b)

print(a*b)

print(a/b)

print(a**2)

print(a**(0.5))

print(a%3) # 求模

print(a//3) # 地板除法

print(a>=2) # torch.ge(a,2) #ge:greater_equal缩写

print((a>=2)&(a<=3))

print((a>=2)|(a<=3))

print(a==5)  # torch.eq(a, 5)

print(torch.sqrt(a))

a = torch.tensor([1.0, 8.0])
b = torch.tensor([5.0, 6.0])
c = torch.tensor([6.0, 7.0])

d = a+b+c
print(d)

print(torch.max(a,b))

print(torch.min(a,b))

x = torch.tensor([2.6, -2.7])

print(torch.round(x)) # 保留整数部分，四舍五入
print(torch.floor(x)) # 保留整数部分，向下规整
print(torch.ceil(x)) # 保留整数部分，向上规整
print(torch.trunc(x)) # 保留整数部分，向0规整

x = torch.tensor([2.6, -2.7])
print(torch.fmod(x,2)) # 做除法取余数
print(torch.remainder(x,2)) # 作除法取剩余的部分，结果恒正

# 幅值裁剪
x = torch.tensor([0.9, -0.8, 100.0, -20.0, 0.7])
y = torch.clamp(x, min=-1, max=1)
z = torch.clamp(x, max=1)
print(x)
print(y)
print(z)

# 向量运算
# 向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另一个向量。
# 统计值
a = torch.arange(1,10).float()
print(a)
print(torch.sum(a))
print(torch.mean(a))
print(torch.max(a))
print(torch.min(a))
print(torch.prod(a)) # 累乘
print(torch.std(a))  # 标准差
print(torch.var(a))  # 方差
print(torch.median(a)) # 中位数

# 指定维度计算统计值
b = a.view(3,3)
print(b)
print(torch.max(b, dim=0))
print(torch.max(b, dim=1))

#cum扫描
a = torch.arange(1,10)

print(torch.cumsum(a, 0))
print(torch.cumprod(a, 0))
print(torch.cummax(a,0).values)
print(torch.cummax(a,0).indices)
print(torch.cummin(a, 0))

# torch.sort和torch.topk可以对张量排序
a = torch.tensor([[9,7,8], [1,3,2], [5,6,4]]).float()
print(torch.topk(a, 2, dim=0), "\n")
print(torch.topk(a, 2, dim=1), "\n")
print(torch.sort(a, dim=1), "\n")

# 利用torch.topk可以在Pytorch中实现KNN算法

# 三、矩阵运算
# 矩阵必须是二维的。类似torch.tensor([1,2,3])这样的不是矩阵。
# 矩阵运算包括：矩阵乘法、矩阵转置、矩阵逆、矩阵求迹、矩阵范数、矩阵行列式、矩阵求特征值、矩阵分解等运算。

# 矩阵乘法
a = torch.tensor([[1,2], [3,4]])
b = torch.tensor([[2,0], [0,2]])
print(a@b) # 等价于torch.matmul(a,b)或torch.mm(a,b)

# 矩阵转置
a = torch.tensor([[1.0, 2], [3, 4]])
print(a.t())

# 矩阵逆，必须为浮点类型
a = torch.tensor([[1.0, 2], [3, 4]])
print(torch.inverse(a))

# 矩阵求trace
a = torch.tensor([[1.0, 2], [3, 4]])
print(torch.trace(a))

# 矩阵行列式
a = torch.tensor([[1.0, 2], [3, 4]])
print(torch.det(a))

# 矩阵特征值和特征向量
a = torch.tensor([[1.0, 2], [-5, 4]], dtype=torch.float)
print(torch.eig(a, eigenvectors=True))

# 两个特征值分别是-2.5+2.7839j, 2.5-2.7839j

# 矩阵QR分解，将一个方阵分解为一个正交矩阵q和上三角矩阵r
# QR分解实际上是对矩阵a实施Schmidt正交化得到q
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
q,r = torch.qr(a)
print(q, "\n")
print(r, "\n")
print(q@r)

# 矩阵svd分解
# svd分解可以将任意一个矩阵分解为一个正交矩阵u，一个对焦矩阵s和一个正交矩阵v.t()的乘积
# svd常用于矩阵压缩和降维
a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

u,s,v = torch.svd(a)
print(u, "\n")
print(s, "\n")
print(v, "\n")

print(u@torch.diag(s)@v.t())

# 利用svd分解可以在Pytorch中实现主成分分析降维


# 四、广播机制
# Pytorch的广播规则和numpy是一样的：
# 1. 如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样。
# 2. 如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为1，那么我们就说这两个张量在该维度上是相容的。
# 3. 如果两个张量在所有维度上都是相容的，它们就能使用广播。
# 4. 广播之后，每个维度的长度将取两个张量在该维度长度的较大值。
# 5. 在任何一个维度上，如果一个张量的长度为1，零一张量长度大于1，那么在该维度上，就好像是对第一个张量进行了复制。
# torch.broadcast_tensor可以将多个张量根据广播规则转换成相同的维度。
a = torch.tensor([1,2,3])
b = torch.tensor([[0,0,0], [1,1,1], [2,2,2]])
print("a", a)
print("b", b)
print(b + a)

a_broad, b_broad = torch.broadcast_tensors(a, b)
print(a_broad, "\n")
print(b_broad, "\n")
print(a_broad + b_broad)
