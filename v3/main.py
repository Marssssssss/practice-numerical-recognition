# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as data_util
import pandas
import matplotlib.pyplot as plt


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
            4、基于损失函数值越小越好这一基础，使用优化器沿着各个参数梯度值的反向进行调整
    """
    def __init__(self):
        super().__init__()

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
        )

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

    def train(self, inputs, targets):  # noqa
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)

        self.counter += 1
        if self.counter % 1000 == 0:
            print("progress: %d/%d" % (self.counter, length))
        self.progress.append(loss.item())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        data = {
            "loss value": pandas.Series(self.progress),
            "train times": pandas.Series(range(len(self.progress))),
        }
        pandas.DataFrame(data).plot(kind="scatter", x="train times", y="loss value", ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()


class MnistDataset(data_util.Dataset):
    """ 用于读取数据的类，参数：
            csv_file - 数据集文件路径

        除了数据集里的数据，__getitem__ 还会返回定制好的目标数据，采用独热编码作为目标
    """
    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        target = torch.zeros(10)
        target[label] = 1.0

        image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values / 255.0)
        return label, image_values, target


mnist_train_dataset = MnistDataset("./data/mnist_train.csv")
mnist_test_dataset = MnistDataset("./data/mnist_test.csv")


def interactively_test_module(c):
    data_len = len(mnist_test_dataset)

    while True:
        row = input("选择你要识别的图片（0-%d，输入 q 退出，输入 a 测试所有测试集并输出正确率）：" % (data_len - 1))

        if row == "q":
            exit(0)

        elif row == "a":
            precisions = []
            correct_count = 0
            all_count = 0

            for target_label, image_values, _ in mnist_test_dataset:
                output = c.forward(image_values)
                if output.argmax() == target_label:
                    correct_count += 1
                all_count += 1
                precisions.append(correct_count / float(all_count))

            data = {
                "index": range(all_count),
                "result": precisions,
            }

            plt.title("final precision = %f" % precisions[-1])
            plt.ylim(0, 1.0)
            plt.scatter(x="index", y="result", data=data, marker=".", alpha=0.1, s=1)
            plt.show()

        elif row.isdigit():
            row_int = int(row)

            if row_int < 0 or row_int >= data_len:
                print("row_index out of range!")
                continue

            target_label, image_values, _ = mnist_test_dataset[row_int]
            img = image_values.detach().numpy().reshape(28, 28)
            output = c.forward(image_values)

            plt.subplot(1, 2, 1)
            plt.title("origin image, label = %d" % target_label)
            plt.imshow(img, interpolation='none', cmap='Blues')

            axes = plt.subplot(1, 2, 2)
            pandas.DataFrame(output.detach().numpy()).plot(kind='bar', legend=False, ylim=(0, 1), ax=axes)

            plt.show()
        else:
            print("unknown input!")


if __name__ == "__main__":
    c = Classifier()
    counter = 0
    length = len(mnist_train_dataset)
    epochs = 3

    for i in range(epochs):
        print("epoch: %d/%d" % (i + 1, epochs))
        for label, image_data_tensor, target_tensor in mnist_train_dataset:
            c.train(image_data_tensor, target_tensor)
        c.counter = 0

    c.plot_progress()
    interactively_test_module(c)
