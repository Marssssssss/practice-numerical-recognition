# -*- coding:utf-8 -*-
import numpy
import scipy.special


class DigitalRecognitionNetwork(object):
    def __init__(self, input_count, hidden_count, final_count, learning_rate):
        self.input_count = input_count  # 输入层节点数
        self.hidden_count = hidden_count  # 隐层节点数
        self.final_count = final_count  # 输出层节点数
        self.learning_rate = learning_rate  # 学习率 

        self.ih_weights = numpy.random.rand(self.hidden_count, self.input_count) - 0.5  # 输入层到隐层权重矩阵
        self.ho_weights = numpy.random.rand(self.final_count, self.hidden_count) - 0.5  # 隐层到输出层权重矩阵

    def recognize(self, inputs_list):
        """ 接收输入，输出数字的判断结果
        @return:
        """
        _, final_outputs = self._recognize(inputs_list)
        return final_outputs

    def _recognize(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # 计算隐层输出
        hidden_inputs = numpy.dot(self.ih_weights, inputs)
        hidden_outputs = scipy.special.expit(hidden_inputs)

        # 计算输出层输出
        final_inputs = numpy.dot(self.ho_weights, hidden_outputs)
        final_outputs = scipy.special.expit(final_inputs)

        return hidden_outputs, final_outputs

    def train(self, inputs_list, targets_list):
        """ 通过预定的样本训练模型
        @return:
        """
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_outputs, final_outputs = self._recognize(inputs_list)

        # 计算隐层误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.ho_weights.T, output_errors)

        # 更新权重
        self.ho_weights += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.ih_weights += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
