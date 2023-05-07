# -*- coding:utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import pandas
import torch

sys.path.insert(0, os.path.normpath(os.path.join(__file__, "../../common")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device:", device)


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

            img = dataset.tensor_to_numpy(image_values).reshape(28, 28)
            output = c.forward(image_values)

            plt.subplot(1, 2, 1)
            plt.title("origin image, label = %d" % target_label)
            plt.imshow(img, interpolation='none', cmap='Blues')

            axes = plt.subplot(1, 2, 2)
            pandas.DataFrame(dataset.tensor_to_numpy(output)).plot(kind='bar', legend=False, ylim=(0, 1), ax=axes)

            plt.show()
        else:
            print("unknown input!")


if __name__ == "__main__":
    import classifier
    import dataset

    mnist_train_dataset, mnist_test_dataset = dataset.get_mnist_dataset(device)

    c = classifier.Classifier(device)
    counter = 0
    length = len(mnist_train_dataset)
    epochs = 3

    for i in range(epochs):
        print("epoch: %d/%d" % (i + 1, epochs))
        for label, image_data_tensor, target_tensor in mnist_train_dataset:
            c.train(image_data_tensor, target_tensor, all_length=length)
        c.counter = 0

    c.plot_progress()
    interactively_test_module(c)
