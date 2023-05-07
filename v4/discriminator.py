# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import pandas
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.to(device)

        # 激活函数
        # Sigmoid - s 型函数，缺点在于大输入值情况下学习效率会大打折扣，因为这里的梯度非常低，处于饱和情况下的梯度消失现象
        # activate_function = nn.Sigmoid
        # ReLU - 线性整流函数，一个打勾的形状，能解决 s 型函数的梯度消失问题，缺点在于在负值区也有梯度消失问题
        # activate_function = nn.ReLU
        # LeakyReLU - 带泄漏的线性整流函数，负值区带有一定的梯度，可以自定义负值区的梯度
        activate_function = (lambda: nn.LeakyReLU(0.02))

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            activate_function(),

            # 标准化，可以将梯度从饱和区拉出来，对于 sigmoid 这种函数形式标准化可以一定程度上减轻梯度消失带来的影响
            nn.LayerNorm(200),

            nn.Linear(200, 1),
            # activate_function()
            nn.Sigmoid()
        ).to(device=self.device)

        # 损失函数
        # MSELoss - 均方误差，适用于回归任务
        # self.loss_function = nn.MSELoss()
        # BCELoss - 二元交叉熵，适用于分类任务，要么错得彻底，要么对得准确，更适用当前模型，但是也有限制，输入得是 [0, 1] 范围内的数，所以如果激活函数本身的输出不在 [0, 1]，要做处理
        self.loss_function = nn.BCELoss()

        # 优化器
        # SGD - 随机梯度下降，这个优化器会根据计算图中每个节点的梯度进行梯度下降计算
        # self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        # Adam - 一种使用了动量概念的优化算法，相比 SGD 更能够避免陷入损失函数的局部最小值，对于鞍点和梯度平滑区都可以快速通过，但是计算时间相当久
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets, all_length=None):  # noqa
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self, ax=None):
        data = {
            "loss value": pandas.Series(self.progress),
            "train times": pandas.Series(range(len(self.progress))),
        }
        max_y = max(self.progress) * 1.1
        pandas.DataFrame(data).plot(kind="scatter", x="train times", y="loss value", ylim=(0, max_y), figsize=(16, 8), alpha=0.1, marker='.',
                                    grid=True,
                                    yticks=numpy.arange(0, max_y, max_y / 10), title="discriminator", ax=ax)
        if ax is None:
            plt.show()
