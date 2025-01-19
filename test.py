from dqn import DQN
from env import Env

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

env = Env()
num_action = env.num_actions()

q_func = DQN(env.num_actions()).to(device)
ckpt = torch.load("episode_34500_score_18.0.pt")
q_func.load_state_dict(ckpt["target_net"])

state = env.reset()
done = False

state, _, done, _ = env.step(1)
# action = 1
while not done:
    q_values = q_func(state.unsqueeze(0).to(device))
    action = torch.argmax(q_values, dim=1).item()

    next_state, reward, done, _ = env.step(action)
    state = next_state
        
