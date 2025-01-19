import torch
import torch.nn as nn
from torchsummary import summary


class DQN(nn.Module):
    def __init__(self, in_channels=4, action_space=4):
        super(DQN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fcn = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, action_space)
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcn(x)
        return x


if __name__ == "__main__":
    in_channels = 4
    summary(
        DQN(in_channels=in_channels, action_space=4),
        (in_channels, 84, 84),
        device="cpu",
    )
