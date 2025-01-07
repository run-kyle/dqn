from env import Env
from dqn import DQN
from ReplayBuffer import ReplayBuffer, Experience
from agent import Agent

import torch
import torch.optim as optim
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_STACKS = 4
NUM_MEMORY = 100

env = Env()
num_action = env.num_actions()

dqn = DQN(NUM_STACKS, num_action)
dqn.to(device)

replay_buffer = ReplayBuffer(NUM_MEMORY)
agent = Agent(
    num_action,
    dqn,
)


epsilon = 1.0
epsilon_decay = 0.000001
epsilon_min = 0.1
num_episodes = 100000


class Trainer:
    def __init__(self, model, replay_buffer):
        self.batch_size = 32
        self.optimizer = optim.RMSprop(model.parameters(), lr=0.00025)
        self.replay_buffer = replay_buffer
        self.model = model
        self.gamma = 0.99

    def update_grad(self):
        batch = self.replay_buffer.sample(self.batch_size)

        batch = Experience(*zip(*batch))

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)

        actions = torch.tensor(batch.action).unsqueeze(1).to(device)
        rewards = torch.tensor(batch.reward).unsqueeze(1).to(device)
        dones = torch.tensor(batch.done).unsqueeze(1).to(device)

        state_action_values = self.model(states).gather(1, actions)
        next_state_values = self.model(next_states).max(1)[0].detach().unsqueeze(1)

        expected_state_action_values = rewards + (
            self.gamma * next_state_values * ~dones
        )

        loss = F.mse_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


trainer = Trainer(dqn, replay_buffer)

num_step = 0
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        replay_buffer.push(Experience(state, action, reward, next_state, done))

        if len(replay_buffer) == NUM_MEMORY:
            trainer.update_grad()

        total_reward += reward
        state = next_state
        epsilon = max(1.0 - num_step * epsilon_decay, epsilon_min)
        num_step += 1

    print(
        f"Episode {episode + 1}, Total Reward: {total_reward}, Buffer Length: {replay_buffer.__len__()}, Epsilon: {epsilon}, Step: {num_step}"
    )


print("done")
