import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, cfg=None):
        super(LeNet5, self).__init__()
        self.cfg = cfg if cfg else [6, 16]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, self.cfg[0], kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.cfg[0], self.cfg[1], kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.cfg[1] * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

