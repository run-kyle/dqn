from env import Env
from dqn import DQN
from ReplayBuffer import ReplayBuffer, Experience
from agent import Agent

import torch
import torch.optim as optim
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_STACKS = 4
NUM_MEMORY = 100000
MIN_MEMORY = 15000

ACTION_REPEAT_FREQ = 4
GRAD_UPDATE_FREQ = 4
TARGET_UPDATE_FREQ = 10000

env = Env()
num_action = env.num_actions()

policy_net = DQN(NUM_STACKS, num_action)
target_net = DQN(NUM_STACKS, num_action)
target_net.load_state_dict(policy_net.state_dict())

policy_net.to(device)
target_net.to(device)


replay_buffer = ReplayBuffer(NUM_MEMORY)
agent = Agent(
    num_action,
    policy_net,
)


epsilon = 1.0
epsilon_decay = 0.000001
epsilon_min = 0.1
num_episodes = 100000


class Trainer:
    def __init__(self, policy_model, target_model, replay_buffer):
        self.batch_size = 32
        self.optimizer = optim.RMSprop(
            policy_model.parameters(), lr=0.00025, alpha=0.95, eps=0.01
        )
        self.replay_buffer = replay_buffer
        self.policy_model = policy_model
        self.target_model = target_model
        self.gamma = 0.99

    def update_grad(self):
        batch = self.replay_buffer.sample(self.batch_size)

        batch = Experience(*zip(*batch))

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)

        actions = torch.tensor(batch.action).unsqueeze(1).to(device)
        rewards = torch.tensor(batch.reward).unsqueeze(1).to(device)
        dones = torch.tensor(batch.done).unsqueeze(1).to(device)

        state_action_values = self.policy_model(states).gather(1, actions)
        next_state_values = self.target_model(next_states).max(1)[0].unsqueeze(1)

        expected_state_action_values = rewards + (
            self.gamma * next_state_values * ~dones
        )

        loss = F.mse_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


trainer = Trainer(policy_net, target_net, replay_buffer)

start_episode = 0
num_step = 0

if False:
    checkpoint = torch.load("episode_0_score_0.0.pt")
    policy_net.load_state_dict(checkpoint["policy_net"])
    target_net.load_state_dict(checkpoint["target_net"])
    trainer.optimizer = checkpoint["optimizer"]
    num_step = checkpoint["num_step"]
    replay_buffer = checkpoint["replay_buffer"]
    start_episode = checkpoint["start_episode"]

for episode in range(start_episode, num_episodes):
    state = env.reset()
    # done = False
    total_reward = 0

    episode_len = 0
    total_loss = 0
    action = 0

    action_select_idx = 0
    gradient_update_idx = 0
    
    state, _, done, _ = env.step(1)
    while not done:
        if num_step % ACTION_REPEAT_FREQ == 0:
            action = agent.select_action(state.to(device), epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(Experience(state, action, reward, next_state, done))

        if len(replay_buffer) > MIN_MEMORY and num_step % GRAD_UPDATE_FREQ == 0:
            total_loss += trainer.update_grad()

        if num_step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        total_reward += reward
        state = next_state
        epsilon = max(1.0 - num_step * epsilon_decay, epsilon_min)
        num_step += 1
        episode_len += 1

    total_loss = total_loss / episode_len

    print(
        f"Episode {episode + 1}, Total Reward: {total_reward}, Buffer Length: {replay_buffer.__len__()}, loss: {total_loss}, eps: {epsilon}"
    )

    if episode % 500 == 0:
        torch.save(
            {
                "policy_net": policy_net.state_dict(),
                "target_net": target_net.state_dict(),
            },
            f"episode_{episode}_score_{total_reward}.pt",
        )
        # torch.save(
        #     {
        #         "policy_net": policy_net.state_dict(),
        #         "target_net": target_net.state_dict(),
        #         "optimizer": trainer.optimizer,
        #         "num_step": num_step,
        #         "replay_buffer": replay_buffer,
        #         "start_episode": episode,
        #     },
        #     f"episode_{episode}_score_{total_reward}.pt",
        # )

print("done")
