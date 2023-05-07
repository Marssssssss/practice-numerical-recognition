# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import pandas
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.to(device)

        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.2),

            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        ).to(self.device)

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model.forward(inputs)

    def train(self, discriminator, inputs, targets, all_length=None):  # noqa
        outputs = self.forward(inputs)
        d_outputs = discriminator.forward(outputs)
        loss = discriminator.loss_function(d_outputs, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self, ax=None):
        data = {
            "loss value": pandas.Series(self.progress),
            "train times": pandas.Series(range(len(self.progress))),
        }
        max_y = max(self.progress) * 1.1
        pandas.DataFrame(data).plot(kind="scatter", x="train times", y="loss value", ylim=(0, max_y), figsize=(16, 8), alpha=0.1,
                                    marker='.', grid=True,
                                    yticks=numpy.arange(0, max_y, max_y / 10), title="generator", ax=ax)
        if ax is None:
            plt.show()
