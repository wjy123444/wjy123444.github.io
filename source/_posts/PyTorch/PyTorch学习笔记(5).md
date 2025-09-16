---
title: PyTorch学习笔记(5)
date: 2025-03-13 20:49:05
tags: PyTorch
categories: 学习
---
# 逻辑斯蒂回归

```python
import torch
import torch.nn.functional as F
x_data= torch.Tensor([[1.0],[2.0],[3.0]])
y_data= torch.Tensor([[0],[0],[1]])

class Logistic_model(torch.nn.Module):
    def __init__(self):
          super(Logistic_model,self).__init__()
          self.linear = torch.nn.Linear(1,1)
  
    def forward(self,x):
        y_prev = F.sigmoid(self.linear(x))
        return y_prev
  
model = Logistic_model()

criterion = torch.nn.BCELoss(reduction= 'sum')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range (1,1000):
    y_prev = model(x_data)
    loss = criterion(y_prev,y_data)
    print(epoch,loss.item())
  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
 
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

```

与上节课实现思路基本一致，区别点如下

* 使用了Logistic Regression将具体的数值映射到0，1表示成功与失败
* 换用BCELoss来表示损失函数，通过比较交叉熵的数值来判断两个分布之间的接近程度，而不是判断具体数值之间的接近程度

本例展示了这套PyTorch编程架构的弹性，可以适用于较多的网状神经网络
