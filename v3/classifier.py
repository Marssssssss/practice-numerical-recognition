# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas
import torch
import torch.nn as nn


class Classifier(nn.Module):
    """ 看源码大致摸出来的步骤：

        __init__ 时：
            1、定义 model，根据 Module.__setattr__ 这个属性会被加到 _modules 里面，供后面运行使用
            2、定义损失函数，这个函数用来做一步损失计算，并允许我们通过 Module 本身的自动微分机制在计算图中传播梯度
            3、定义优化器，损失函数计算完梯度后，优化器扫描整个 module 里面的每一个参数，并根据梯度和优化器本身的算法调整参数

            Linear - 全连接层
            Sigmoid - 每个节点的激活函数，在上一层计算到这层之后，计算的结果经过激活函数进一步计算该节点最新的输出值

        train 时：
            1、使用 forward 前向传播计算结果
            2、用损失函数进一步计算损失，并产生各个参数到结果的传播关系
            3、清理梯度并反向传播计算每个参数的梯度
            4、基于损失函数的值越小越好，使用优化器沿着各个参数梯度值的反向进行调整
    """

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
        activate_function = (lambda: nn.LeakyReLU(0.2))

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            activate_function(),

            # 标准化，可以将梯度从饱和区拉出来，对于 sigmoid 这种函数形式标准化可以一定程度上减轻梯度消失带来的影响
            nn.LayerNorm(200),

            nn.Linear(200, 10),
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
        self.optimiser = torch.optim.Adam(self.parameters())

        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets, all_length=None):  # noqa
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)

        self.counter += 1
        if self.counter % 1000 == 0:
            print("classifier progress:",
                  "%d/%d" % (self.counter, all_length) if all_length is not None else self.counter)
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        data = {
            "loss value": pandas.Series(self.progress),
            "train times": pandas.Series(range(len(self.progress))),
        }
        pandas.DataFrame(data).plot(kind="scatter", x="train times", y="loss value", ylim=(0, 1.0), figsize=(16, 8),
                                    alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
