---
title: PyTorch学习笔记(1)
date: 2025-03-07 14:02:37
tags: PyTorch
categories: 学习
---
# 线性模型

传送门  [Pytorch深度学习教程](https://www.bilibili.com/video/BV1Y7411d7Ys?vd_source=3b3f552b4d44feecf34501451b229f5e&spm_id_from=333.788.videopod.episodes&p=2) 本文借鉴CSDN专栏 [PyTorch 深度学习实践](https://blog.csdn.net/bit452/category_10569531.html)

```python
import numpy as np
import matplotlib.pyplot as plt
 
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
 
def forward(x):
    return x*w
 
 
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
 
# 穷举法
w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)
  
plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()  
```

## 代码说明

1. 函数forward中的w在后续的for循环中传入
2. np.arange numpy.arange([start, ]stop, [step, ]dtype=None)

* **start**（可选）：序列的起始值，默认为 `0`。
* **stop** ：序列的结束值（ **不包含该值本身** ）。
* **step**（可选）：步长（间隔），默认为 `1`。
* **dtype**（可选）：指定输出数组的数据类型（如 `int`, `float` 等）。
* 不包括结束值

---

3. zip函数在这里的作用是配对基本元素
