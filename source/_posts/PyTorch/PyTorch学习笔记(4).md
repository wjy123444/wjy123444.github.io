---
title: PyTorch学习笔记(4)
date: 2025-03-11 14:02:37
tags: PyTorch
categories: 学习
---
# 利用PyTorch实现线性回归

传送门  [Pytorch深度学习教程](https://www.bilibili.com/video/BV1Y7411d7Ys?vd_source=3b3f552b4d44feecf34501451b229f5e&spm_id_from=333.788.videopod.episodes&p=2) 本文借鉴CSDN专栏 [PyTorch 深度学习实践](https://blog.csdn.net/bit452/category_10569531.html)

本节共分为四步

## Prepare dataset

后续有方法，现在目前用简单手打代替

```python
x_data=torch.tensor([[1.0],[2.0],[3.0]])
y_data=torch.tensor([[2.0],[4.0],[6.0]])
```

## Design model

```python
class LinearModle(torch.nn.Module):
     def __init__(self):
        super(LinearModle,self).__init__() 
        self.linear=torch.nn.Linear(1,1)
    
     def forward(self,x):
         y_prev = self.linear(x)
         return y_prev

model = LinearModle()
```

封装好类，使用仅需要实例化即可

## Construct loss and optimizer

```python
criterion = torch.nn.MSELoss(reduction= 'sum')
optimiser = torch.optim.SGD(model.parameters(),lr = 0.01)
```

torch.nn.MSEloss用于计算损失，参数默认是mean取平均值sum参数求和

优化器中传入需要学习的参数，lr=0.01代表学习率 用于实现反向传播和随机梯度下降算法

## Training cycle (forward,backward,update)

```python
for epoch in range(1,100):
    y_prev = model(x_data)
    loss = criterion(y_prev,y_data)
    print (epoch,loss.item())
    print('w=',model.linear.weight.item())
    print('b=',model.linear.bias.item())
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
```

这段需要注意的是 每轮需要先将grad清零 否则会累加
