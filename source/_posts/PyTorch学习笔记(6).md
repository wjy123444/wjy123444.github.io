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

```

```
