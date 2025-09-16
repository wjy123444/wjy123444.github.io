---
title: PyTorch学习笔记(6)
date: 2025-03-14 12:33:16
tags: PyTorch
categories: 学习
---
# Multiple Dimension Input

1. 对于线性模型的处理，将每个维度的x输入值乘不同的权重再加上偏移量
2. 对于PyTorch支持的函数，是直接作用于1*N矩阵中的每一个值的
3. 尽量把运算转化为矩阵向量运算，可以利用GPU的并行计算的能力提高运算速度
4. **Linear（8，1）** 输入维度8维输出维度1维
5. Linear（8，2）-> Linear (2,1) 多层神经网络 矩阵是空间变换的函数

```python
import torch 
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('diabetes.csv.gz',delimiter=',',dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()
      
    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

criterion  = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

epoch_list = []
loss_list = []
for epoch in range(1,100):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())
  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(epoch_list,loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```
