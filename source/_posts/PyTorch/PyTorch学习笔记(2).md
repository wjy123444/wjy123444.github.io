---
title: PyTorch学习笔记(2)
date: 2025-03-10 19:48:33
tags: PyTorch
categories: 学习
---
# 梯度下降算法

传送门  [Pytorch深度学习教程](https://www.bilibili.com/video/BV1Y7411d7Ys?vd_source=3b3f552b4d44feecf34501451b229f5e&spm_id_from=333.788.videopod.episodes&p=3) 本文借鉴CSDN专栏 [PyTorch 深度学习实践](https://blog.csdn.net/bit452/category_10569531.html)

```python
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

def forward (x):
  return x*w

def cost(xs,ys):
    loss=0
    for x,y in zip(xs,ys):
        y_prev = forward (x)
        loss += (y_prev - y) **2
    return loss / len(xs)

def grad(xs,ys):
    g=0
    for x,y in zip(xs,ys):
       y_prev = forward(x)
       g+=2*x*(y_prev - y)
    return g / len(xs)
epoch_list = []
loss_list = []
w = 1 
for epoch in range(1,100):
    cost_val = cost(x_data,y_data)
    g=grad(x_data,y_data)
    w-=0.01 * g
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    loss_list.append(cost_val)
plt.plot(epoch_list,loss_list)
plt.xlabel('epoch')
plt.ylabel('loss') 
plt.show()   
```

注：梯度下降算法 无法适用于存在鞍点函数图像，一旦函数图像出现不增不减的情况，w将无法因为训练轮次的增加而改变。并且当靠近局部最优点的时候，将无法找到全局最优点。

在真实的深度学习训练中，常采用随机梯度下降法

```python

import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

def forward(x):
    return x * w

def loss(x,y):
   y_prev = forward (x)
   return (y_prev - y) **2

def grad(x,y):
   y_prev = forward(x)
   return 2*x*(y_prev -y)

epoch_list = []
loss_list = []
w = 4
for epoch in range(1,100):
    for x,y in zip(x_data,y_data):
        g=grad(x,y)
        loss_val=loss(x,y)
        w-=0.01*g
    print("epoch=",epoch,"w=",w,"loss=",loss_val)
    epoch_list.append(epoch)
    loss_list.append(loss_val)
plt.plot(epoch_list,loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```

1. 随机梯度算法加大了样本的噪声 使得算法可能面对鞍点仍然可以训练
2. 计算单个样本的损失和梯度 减少了计算量加快了效率
3. 实际的深度学习算法常采用将少量数据打包成mini-batch 采用随机梯度下降的算法的方式是两种算法的折中PyTorch学习笔记(2)选项
