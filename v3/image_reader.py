# -*- coding:utf-8 -*-
# 使用方式：直接运行，有交互内容

def read_images(path):
    if path is None:
        return

    import pandas
    import matplotlib.pyplot as plt

    data = pandas.read_csv(path, header=None)
    data_len = data.shape[0]

    while True:
        row = input("选择你要查看的图片（0-%d，输入 q 退出）：" % (data_len - 1))

        if row == "q":
            exit(0)
        elif row.isdigit():
            row_int = int(row)

            if row_int < 0 or row_int >= data_len:
                print("row_index out of range!")
                continue

            row_data = data.iloc[row_int]
            label = row_data[0]
            img = row_data[1:].values.reshape(28, 28)
            plt.title("label = %d" % label)
            plt.imshow(img, interpolation='none', cmap='Blues')
            plt.show()
        else:
            print("unknown input!")


if __name__ == "__main__":
    data_type = input("选择你要查看的数据：\n1、训练集\n2、测试集\n输入数字选择：")
    paths = {
        "1": "./data/mnist_train.csv",
        "2": "./data/mnist_test.csv",
    }
    read_images(paths.get(data_type))
