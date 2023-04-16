# -*- coding:utf-8 -*-
import numpy
import scipy.special


class DigitalRecognitionNetwork(object):
    """ 支持任意层，每层任意节点数量配置 """

    def __init__(self, layer_node_counts, learning_rate):
        self.layer_node_counts = layer_node_counts  # 列表，按序存储每一层的节点数，列表头和尾分别表示输入层和输出层，中间为隐层
        self.layer_count = len(self.layer_node_counts)

        if self.layer_count < 3:
            raise Exception("Network layer count should not be less than 3!")
            return

        self.weights = [numpy.random.rand(self.layer_node_counts[i + 1], self.layer_node_counts[i]) - 0.5 for i in range(self.layer_count - 1)]
        self.outputs = []

        self.learning_rate = learning_rate  # 学习率

    def recognize(self, input_data):
        """ 识别，返回识别结果
        @return:
        """
        self._recognize(input_data)
        return self.outputs[-1].argmax()

    def _recognize(self, input_data):
        """ 识别过程，仅更新对象内状态
        @return:
        """
        self.outputs = [numpy.array(input_data, ndmin=2).T]

        # 正向传播
        for i in range(self.layer_count - 1):
            next_outputs = scipy.special.expit(numpy.dot(self.weights[i], self.outputs[i]))
            self.outputs.append(next_outputs)

    def train(self, input_data, target_output):
        """ 训练，先正向传播，然后根据状态反向传播更新权重
        @return:
        """
        self._recognize(input_data)
        target_output = numpy.array(target_output, ndmin=2).T

        # 计算误差
        output_errors = [target_output - self.outputs[-1]]  # 误差列表，按序依次存储输出层到第一层隐层中间所有层的误差
        for i in range(self.layer_count - 2):
            output_errors.append(numpy.dot(self.weights[self.layer_count - i - 2].T, output_errors[i]))

        # 更新权重
        for i in range(self.layer_count - 1):
            next_output = self.outputs[i + 1]  # 权重连接目标层输出
            last_output = self.outputs[i]  # 权重连接来源层输出
            error = output_errors[self.layer_count - 2 - 1]  # 权重连接目标层的误差
            self.weights[i] += self.learning_rate * numpy.dot((output_errors[self.layer_count - 2 - i] * next_output * (1.0 - next_output)), numpy.transpose(last_output))
