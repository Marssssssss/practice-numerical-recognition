# -*- coding:utf-8 -*-
import digital_recognition
import numpy
import sys


def train(nn):
    train_sample_path = sys.argv[1]
    try:
        with open(train_sample_path, "r+") as f:
            samples = f.readlines()
    except:
        print("open %s failed!" % train_sample_path)
        exit()
    
    
    for record in samples:
        all_values = record.split(",")
        inputs = numpy.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
        targets = numpy.zeros(10) + 0.01
        targets[int(all_values[0])] = 0.99
        
        nn.train(inputs, targets)


def test(nn):
    test_sample_path = sys.argv[2]
    try:
        with open(test_sample_path, "r+") as f:
            test_samples = f.readlines()
    except:
        print("open %s failed!" % test_sample_path)
        exit()

    correct_count = 0
    mistake_count = 0
    for record in test_samples:
        all_values = record.split(",")
        inputs = numpy.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01

        result = nn.recognize(inputs)
        if int(all_values[0]) == result:
            correct_count += 1
        else:
            mistake_count += 1
    print("precision:", correct_count / (correct_count + mistake_count))


if __name__ == "__main__":
    nn = digital_recognition.DigitalRecognitionNetwork([784, 1000, 10], 0.3)

    train(nn)
    test(nn)
