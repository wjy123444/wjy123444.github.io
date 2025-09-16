---
title: PyTorch学习笔记(8)
date: 2025-03-15 17:33:16
tags: PyTorch
categories: 学习
---
# Softmax Classifier

* 我们希望神经网络输出一个分布，所以我们希望输出的不同类别之间是有竞争性的
* Softmax计算公式，保证每种分类大于零，且和为1

$$
P(y = i) = \frac{e^{Z_i}}{\sum_{j=0}^{K-1} e^{Z_j}}, \quad i \in \{0, \ldots, K-1\}
$$

* CrossEntropyLoss（）不需要做激活
* 将图像中的0-255的像素映射到0-1成为一个矩阵
* transfroms   convert the pil Image to Tensor
* 图像张量 多通道 CWH
*
