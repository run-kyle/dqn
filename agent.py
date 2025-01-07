import torch
import torch.nn.functional as F
import random


class Agent:
    def __init__(self, action_space, policy_net):
        self.num_action = action_space
        self.q_net = policy_net

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.num_action))
        else:
            with torch.no_grad():
                # a = self.q_net.parameters()
                # b = a.__next__()
                # print(b[0][-1])
                q_values = self.q_net(state.unsqueeze(0))
                return q_values.argmax().item()
