正好这段时间 `AI` 算是又火了一波，顺势了解了一波，跟着《Python 神经网络编程》这本书写了个简单的三层神经网络做单个手写数字识别，借此了解下 `nn` 的思想。训练集和测试集选用 `MNIST` 数据集，训练集使用 `100` 个样本，测试集使用 `10` 个样本。 


- v1：原始的三层网络，最基本的输入层 + 隐层 + 输出层，初始权重完全用随机来做；
- v2: 在 v1 的基础上扩展到可以指定层数以及每层的节点数量，初始权重完全用随机来做；
