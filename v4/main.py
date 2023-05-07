# -*- coding:utf-8 -*-
import itertools
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.normpath(os.path.join(os.path.abspath(__file__), "../../common")))

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device:", device)


def generate_random_seed(size):
    random_data = torch.randn(size, device=device)
    return random_data


def interactively_test_generator(generator):
    while True:
        input_value = input("输入 q 退出，输入 a 批量生成 9 个图）：")

        if input_value == "q":
            exit(0)

        elif input_value == "a":
            for i in range(9):
                input_value = generate_random_seed(100)
                plt.subplot(1, 9, i + 1)
                plt.title(i + 1)
                plt.imshow(dataset.tensor_to_numpy(generator.forward(input_value)).reshape(28, 28), interpolation='none', cmap='Blues')
            plt.show()

        else:
            print("unknown input!")


if __name__ == "__main__":
    import discriminator
    import generator
    import dataset

    mnist_train_dataset, mnist_test_dataset = dataset.get_mnist_dataset(device)

    discriminator = discriminator.Discriminator(device)
    generator = generator.Generator(device)

    all_length = len(mnist_train_dataset) + len(mnist_test_dataset)
    epochs = 4

    for epoch in range(epochs):
        print("epochs: %d/%d" % (epoch + 1, epochs))

        counter = 0
        for _, image_data_tensor, _ in itertools.chain(mnist_train_dataset, mnist_test_dataset):
            counter += 1
            discriminator.train(image_data_tensor, torch.tensor([1.0], device=device))
            discriminator.train(generator.forward(generate_random_seed(100)).detach(), torch.tensor([0.0], device=device))

            generator.train(discriminator, generate_random_seed(100), torch.tensor([1.0], device=device))

            if counter % 1000 == 0:
                print("progress: %d/%d" % (counter, all_length))

    _, axs = plt.subplots(1, 2, figsize=(6, 4))
    discriminator.plot_progress(ax=axs[0])
    generator.plot_progress(ax=axs[1])

    plt.show()

    interactively_test_generator(generator)
