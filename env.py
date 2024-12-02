import gym
import torch
from torchvision import transforms
from collections import deque


class Env:
    def __init__(self):
        self.env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=16)
        # self.env = gym.make("ALE/Breakout-v5")
        self.env.reset()
        self.action_space = self.env.action_space
        self.memory = deque([], maxlen=4)
        self.memory.append(torch.zeros(1, 84, 84, dtype=torch.float32))
        self.memory.append(torch.zeros(1, 84, 84, dtype=torch.float32))
        self.memory.append(torch.zeros(1, 84, 84, dtype=torch.float32))
        self.memory.append(torch.zeros(1, 84, 84, dtype=torch.float32))

        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
            ]
        )

    def reset(self):
        obs, _ = self.env.reset()
        self.memory.append(self.tf(obs))
        return torch.stack(list(self.memory)).squeeze(1)

    def step(self, action):
        obs, reward, done, _, _ = self.env.step(action)
        self.memory.append(self.tf(obs))
        return torch.stack(list(self.memory)).squeeze(1), reward, done

    def num_actions(self):
        return self.env.action_space.n

    def close(self):
        self.close()


if __name__ == "__main__":
    done = False
    env = Env()
    while not done:
        action = env.action_space.sample()
        obs, reward, done = env.step(action)

    env.close()
