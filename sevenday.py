# 2-3, 动态计算图
# 本节我们将介绍Pytorch的动态计算图
# 包括：
# 动态计算图简介
# 计算图中的Function
# 计算图和反向传播
# 叶子节点和非叶子节点
# 计算图在TensorBoard中的可视化

# 一、动态计算图简介
# Pytorch的计算图由节点和边组成，节点表示张量或者Function，边表示张量和Function之间的依赖关系。
# Pytorch中的计算图是动态图。这里的动态主要由两重含义。
# 第一层含义是：计算图的正向传播是立即执行的。无需等待完整的计算图创建完毕，每条语句都会在计算图中动态节点和边，并立即执行正
# 向传播得到计算结果。
# 第二层含义是：计算图在反向传播后立即销毁。下次调用需要重新构建计算图。如果在程序中使用了backward方法执行了反向传播，或者利用
# torch.autograd.grad方法计算了梯度，那么创建的计算图会被立即销毁，释放存储空间，下次调用需要创建。

# 1. 计算图的正向传播是立即执行的。
import torch
w = torch.tensor([[3.0, 1.0]], requires_grad=True)
b = torch.tensor([[3.0]], requires_grad=True)
X = torch.randn(10,2)
Y = torch.randn(10,1)
Y_hat = X@w.t() + b # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
loss = torch.mean(torch.pow(Y_hat-Y, 2))

print(loss.data)
print(Y_hat.data)

# 2.计算图在反向传播后立即销毁。
import torch
w = torch.tensor([[3.0, 1.0]], requires_grad=True)
b = torch.tensor([[3.0]], requires_grad=True)
X = torch.randn(10, 2)
Y = torch.randn(10, 1)
Y_hat = X@w.t() + b # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
loss = torch.mean(torch.pow(Y_hat-Y, 2))

# 计算图在反向传播后立即销毁，如果需要保留计算图，需要设置retain_graph = True
loss.backward()  # loss.backward(retain_graph=True)

# loss.backward() # 如果再次执行反向传播将会报错

# 二、计算图中的Function
# 计算图中的张量我们已经比较熟悉了，计算图中的另外一个节点是Function，实际上就是Pytorch中各种对张量操作的函数。
# 这些Function和我们Python中的函数有一个较大的区别，那就是它同时包含正向计算逻辑和反向传播的逻辑。
# 我们可以通过继承torch.autograd.Function来创建这种支持反向传播的Function

class MyReLU(torch.autograd.Function):
    # 正向传播逻辑，可以用ctx存储一些值，供反向传播使用。
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    # 反向传播逻辑
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

w = torch.tensor([[3.0, 1.0]], requires_grad=True)
b = torch.tensor([[3.0]], requires_grad=True)
X = torch.tensor([[-1.0, 1.0], [1.0, 1.0]])
Y = torch.tensor([[2.0, 3.0]])

relu = MyReLU.apply # relu现在也可以具有正向传播和反向传播功能
Y_hat = relu(X@w.t() + b)
loss = torch.mean(torch.pow(Y_hat-Y, 2))

loss.backward()

print(w.grad)
print(b.grad)

# Y_hat的梯度函数即是我们自己所定义的MyReLU.backward
print(Y_hat.grad_fn)

# 三、计算图与反向传播
# 了解Function的功能，我们可以简单地理解一下反向传播的原理和过程。理解该部分原理需要一些高等数学中求导链式法则的基础知识。


# 五、计算图在TensorBoard中的可视化
# 可以利用torch.utils.tensorboard将计算图导出到TensorBoard进行可视化。
