---
title: PyTorch学习笔记(7)
date: 2025-03-15 12:33:16
tags: PyTorch
categories: 学习
---
# Dataset and Dataloader

```python
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
  
    def __len__(self):
        return self.len
  
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=30, num_workers=0)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
      
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 降低学习率

epoch_list = []
loss_list = []

if __name__ == '__main__':
    for epoch in range(1, 100):  # 增加训练轮数
        epoch_loss = 0.0  # 记录当前 epoch 的总损失
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
          
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # 累加当前 batch 的损失
      
        # 计算当前 epoch 的平均损失
        avg_loss = epoch_loss / len(train_loader)
        epoch_list.append(epoch)
        loss_list.append(avg_loss)
        print(f'Epoch [{epoch}], Loss: {avg_loss:.4f}')

    # 绘制损失曲线
    plt.plot(epoch_list, loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()
```
