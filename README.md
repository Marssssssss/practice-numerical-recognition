正好这段时间 `AI` 算是又火了一波，顺势了解了一波，跟着《Python 神经网络编程》这本书写了个简单的三层神经网络做单个手写数字识别，借此了解下 `nn` 的思想。训练集和测试集选用 `MNIST` 数据集，训练集使用 `100` 个样本，测试集使用 `10` 个样本。 



虽然最早是跟着书本写的三层网络，但实际上为了探究网络架构和参数的影响，还是打算做一些拓展的。主要是几个拓展点：



- 训练数据分层采样、`n` 折交叉验证，重复训练看下效果；
- 扩大样本数再测测；



这些拓展点会慢慢实现进去，同时做最终生成的数据对比。同样，对应的代码版本：



- v1：原始的三层网络，最基本的输入层 + 隐层 + 输出层，初始权重完全用随机来做；
- v2: 在 v1 的基础上扩展到可以指定层数以及每层的节点数量，初始权重完全用随机来做；



结果：



- v1：准确度 (0.6, 0.5, 0.6, 0.5, 0.5)；
- v2 用一个列表表示层数和每层节点数量，结果分别如下：
  - [784, 100, 10]：准确度（0.5, 0.5, 0.6, 0.6, 0.5）
  - [784, 100, 100, 10]：准确度（0.5, 0.5, 0.6, 0.6, 0.5）
  - [784, 100, 100, 100, 10]：准确度（0.2, 0.2, 0.3, 0.2, 0.2）
  - [784, 100, 100, 100, 100, 10]：准确度（0.0, 0.0, 0.0, 0.0, 0.1）
  - [784, 5, 5, 10]: 准确度（0.1, 0.1, 0.1, 0.1, 0.1）
  - [784, 5, 5, 5, 5, 5, 10]：准确度（0.1, 0.1, 0.1, 0.1, 0.1）
  - [784, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10]：准确度（0.1, 0.1, 0.1, 0.1, 0.1）
  - [784, 1000, 10]：准确度（0.7, 0.6, 0.6, 0.7, 0.6）
  - [784, 10000, 10]：准确度（0.7, 0.6, 0.6, 0.7, 0.6）
  - [784, 10000, 10000, 10]：准确度（0.1, 0.2, 0.2, 0.1, 0.0）
  - [784, 100000, 10]：准确度（0.7, 0.6, 0.6, 0.7, 0.6）



分析：

- v2：层数过高或者节点数过高都可能导致 `过拟合`，而过少又可能导致 `欠拟合`，上面的试验可以看出，一层隐层且节点数 `1000` 的泛化性能和训练性能都相对要好一些。