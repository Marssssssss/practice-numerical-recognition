# -*- coding:utf-8 -*-
import os

import pandas
import torch
import torch.utils.data as data_util


class MnistDataset(data_util.Dataset):
    """ 用于读取数据的类，参数：
            csv_file - 数据集文件路径

        除了数据集里的数据，__getitem__ 还会返回定制好的目标数据，采用独热编码作为目标
    """

    def __init__(self, csv_file, device):
        self.data_df = pandas.read_csv(csv_file, header=None)
        self.device = device

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        target = torch.zeros(10, device=self.device)
        target[label] = 1.0

        image_values = torch.tensor(self.data_df.iloc[index, 1:].values / 255.0, device=self.device,
                                    dtype=torch.float32)
        return label, image_values, target


def tensor_to_numpy(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().numpy()


def get_mnist_dataset(device):
    return MnistDataset(os.path.join(__file__, "../../mnist_dataset/mnist_train.csv"), device), \
           MnistDataset(os.path.join(__file__, "../../mnist_dataset/mnist_test.csv"), device)
