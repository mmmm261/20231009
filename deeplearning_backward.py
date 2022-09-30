#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 17:37
# @Author  : William Baker
# @FileName: SGD_torch.py
# @Software: PyCharm
# @Blog    : https://blog.csdn.net/weixin_43051346

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

#tensor中包含data(w)和grad(loss对w求导)
w = torch.tensor([1.0])     # w的初值为1.0
w.requires_grad = True      # 需要计算梯度
print("w_data:", w.data)
# 学习率 learning rate
lr = 0.01

def forward(x):
    return x * w     # x和w都是tensor类型，数乘

epoch_list = []
cost_list = []

for epoch in range(2):

    for x, y in zip(x_data, y_data):

        y_pred = forward(x)
        l = (y_pred - y) ** 2
        l.backward()

        print('\tgrad:', "input:", x, "pred:", y_pred.detach().numpy()[0], "label", y, "loss:",
              l.detach().numpy()[0], " loss对w求偏导:",w.grad.item())


        # w -= lr * grad_val        # w = w - lr * gradient(w)   梯度下降的核心所在
        # print(w.data.requires_grad)    # False
        w.data = w.data - lr * w.grad.data       # 权重更新时，需要用到标量，注意grad也是一个tensor   # w.grad.item()是等价于 w.grad.data的，都是不建立计算图
        # print(w.data.requires_grad)    # False

        w.grad.data.zero_()     # after update, remember set the grad to zero     # 把权重里面的梯度数据清0，不然就变成了梯度累加

    epoch_list.append(epoch)
    cost_list.append(l.detach().numpy())
    # print('progress:', epoch, l.item())
    print('Progress: Epoch {}, loss:{}'.format(epoch, l.item()))    # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）
                                                                          # Progress: Epoch 99, loss:9.094947017729282e-13
    # print('Progress-: Epoch {}, loss:{}'.format(epoch, l.data.item()))  # Progress-: Epoch 99, loss:9.094947017729282e-13

print("***************************训练结束***************************\n")
# print("predict (after training)", 4, forward(4).item())
print('训练后的输入值x：{}, 训练后的预测值：{}'.format(4, forward(4).item()))

# 绘图
plt.plot(epoch_list, cost_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
