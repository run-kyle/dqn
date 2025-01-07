import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from ReplayBuffer import Experience

device = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    def __init__(
        self, action_space, policy_net, target_net, replay_buffer, batch_size, gamma, lr
    ):
        self.action_space = action_space
        self.policy_net = policy_net
        self.target_net = target_net
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.action_space))
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0).to(device))
                return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)

        batch = Experience(*zip(*batch))

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)

        actions = torch.tensor(batch.action).unsqueeze(1).to(device)
        rewards = torch.tensor(batch.reward).unsqueeze(1).to(device)
        dones = torch.tensor(batch.done).unsqueeze(1).to(device)

        state_action_values = self.policy_net(states).gather(1, actions)

        next_state_values = self.target_net(next_states).max(1)[0].detach()
        expected_state_action_values = rewards + (
            self.gamma * next_state_values * ~dones
        )

        loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
