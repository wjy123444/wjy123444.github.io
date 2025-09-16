---
title: PyTorch学习笔记(3)
date: 2025-03-10 20:41:20
tags: PyTorch
categories: 学习
---
传送门  [Pytorch深度学习教程](https://www.bilibili.com/video/BV1Y7411d7Ys?vd_source=3b3f552b4d44feecf34501451b229f5e&spm_id_from=333.788.videopod.episodes&p=3) 本文借鉴CSDN专栏 [PyTorch 深度学习实践](https://blog.csdn.net/bit452/category_10569531.html)

# 反向传播算法

这节引出了一个重要概念 Tensor 网上相关文档很多 但是眼花缭乱 本蒟蒻看的云里雾里

本蒟蒻的理解目前主要分为三点

1. Tensor是一个多维数组结构，可以表示数值，向量，矩阵甚至更高维的数据
2. Tensor具有自动求导求梯度的特性
3. Python 的标量值（`int`、`float`、`bool`）在与 Tensor 运算时， **会自动转换为与 Tensor 相同数据类型（dtype）的 Tensor** 。

```python
import torch

# 1. 创建 Tensor
a = torch.tensor([1, 2, 3])          # 从列表创建
b = torch.rand(2, 3)                # 创建 2x3 的随机数矩阵（范围[0,1)）
c = torch.zeros(3)                  # 创建全零向量

print("Tensor a:", a)
print("随机矩阵 b:\n", b)
print("全零向量 c:", c)

# 2. 查看基本属性
print("\nTensor属性示例:")
print("形状（shape）:", b.shape)      # 输出: torch.Size([2, 3])
print("数据类型（dtype）:", b.dtype)  # 输出: torch.float32
print("存储设备（device）:", b.device) # 默认是 CPU

# 3. 基础数学运算
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

add = x + y   # 逐元素相加
mul = x * 2   # 逐元素乘标量

print("\n运算结果:")
print("加法:", add)  # tensor([5., 7., 9.])
print("乘标量:", mul) # tensor([2., 4., 6.])

# 4. 形状操作
matrix = torch.rand(2, 4)
reshaped = matrix.view(4, 2)  # 改变形状为 4x2
flattened = matrix.flatten()  # 展平为 1D

print("\n原始矩阵:\n", matrix)
print("变形后:\n", reshaped)
print("展平后:", flattened)

# 5. 自动微分简单示例
w = torch.tensor(3.0, requires_grad=True)  # 启用梯度跟踪
y = w ** 2
y.backward()  # 自动计算梯度
print("\n梯度计算:")
print("dy/dw:", w.grad)  # 输出: tensor(6.)
```

运行结果如下：![](https://fastly.jsdelivr.net/gh/lazysqrt2/blog-img/20250310214210090.png)

反向传播算法实例

```python
import torch
import matplotlib.pyplot as plt
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = torch.tensor(1.0)
w.requires_grad = True
def forward(x):
    return x * w
def loss(x,y):
   y_prev = forward (x)
   return (y_prev-y) ** 2

epoch_list = []
loss_list = []

for epoch in range(1,100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward()
        print (x,y,w.grad.item())
        w.data = w.data - 0.01*w.grad.data
        w.grad.data.zero_() # after update, remember set the grad to zero
    epoch_list.append(epoch)
    loss_list.append(l.item())
    print('progress:', epoch, l.item())
plt.plot(epoch_list,loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```

需要注意的点

* 注意什么时候使用.data来避免参与梯度计算图
* 什么时候使用.item用于获取Tensor的标量(仅适用于单元素Tensor)
