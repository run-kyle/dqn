import gym
import torch
import numpy as np
from torchvision import transforms
from collections import deque
from gym.wrappers.record_video import RecordVideo


def save_image(obs, img_path="temp.png"):
    to_pil = transforms.ToPILImage()
    to_pil(obs).save(img_path)


def crop_img(img):
    return img
    # return img[34:194, :, :]


class Env:
    def __init__(self, num_stack=4, width=84, height=84):
        # self.env = RecordVideo(
        #     gym.make("BreakoutDeterministic-v4", render_mode="rgb_array"),
        #     video_folder=".",
        #     episode_trigger=lambda x: x % 100 == 0,
        # )
        # self.env = gym.make("ALE/Breakout-v5", render_mode="human")
        self.env = gym.make("BreakoutDeterministic-v4", render_mode="human")
            
        self.env.reset()
        self.action_space = self.env.action_space
        self.num_stack = num_stack

        self.memory = deque(
            [],
            maxlen=num_stack,
        )

        self.prev_obs = None

        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                lambda x : transforms.functional.crop(x, 25, 8, 180, 144),
                transforms.Resize((height, width), 0),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x*255.0).byte())
            ]
        ) 

    def reset(self):
        obs, _ = self.env.reset()
        self.prev_obs = obs
        
        [self.memory.append(torch.zeros((1, 84, 84))) for _ in range(self.num_stack)]
        
        # self.step(1)

        return torch.cat(list(self.memory))

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        
        
        max_pixel_img = np.maximum(obs, self.prev_obs)
        self.prev_obs = obs
        self.memory.append(self.tf(crop_img(max_pixel_img)))
    
        # self.memory.append(self.tf(crop_img(obs)))
        
          
        return torch.cat(list(self.memory)), reward, done, info

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
