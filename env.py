import gym
import torch
from torchvision import transforms
from collections import deque
from gym.wrappers.record_video import RecordVideo
from gym.wrappers import monitoring


class Env:
    def __init__(self, num_stack=4, width=84, height=84):
        self.env = RecordVideo(
            gym.make("ALE/Breakout-v5", render_mode="rgb_array"),
            video_folder=".",
            episode_trigger=lambda x: x % 10 == 0,
        )

        self.env.reset()
        self.action_space = self.env.action_space
        self.memory = deque(
            [torch.zeros(1, height, width, dtype=torch.float32)] * num_stack,
            maxlen=num_stack,
        )

        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((height, width)),
                transforms.ToTensor(),
            ]
        )

    def reset(self):
        obs, _ = self.env.reset()
        self.memory.append(self.tf(obs))
        return torch.cat(list(self.memory))

    def step(self, action):
        obs, reward, done, _, _ = self.env.step(action)
        self.memory.append(self.tf(obs))
        return torch.cat(list(self.memory)), reward, done

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
