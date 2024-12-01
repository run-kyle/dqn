import torch
import torch.nn as nn
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=True):
        super(CNN, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.activate = nn.ReLU() if activate else nn.Identity()

    def forward(self, x):
        x = self.cnn(x)
        x = self.activate(x)
        return x


class FCN(nn.Module):
    def __init__(self, in_features, out_features, activate=True):
        super(FCN, self).__init__()
        self.fcn = nn.Linear(in_features, out_features)
        self.activate = nn.ReLU() if activate else nn.Identity()

    def forward(self, x):
        x = self.fcn(x)
        x = self.activate(x)
        return x


class DQN(nn.Module):
    def __init__(self, in_channels, action_space):
        super(DQN, self).__init__()
        self.cnn = nn.Sequential(CNN(in_channels, 16, 8, 4), CNN(16, 32, 4, 2))
        self.fc = nn.Sequential(FCN(32 * 9 * 9, 256), FCN(256, action_space, False))

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    in_channels = 4
    summary(DQN(in_channels=in_channels, action_space=4), (in_channels, 84, 84))
