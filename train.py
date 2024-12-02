from env import Env
from dqn import DQN
from ReplayBuffer import ReplayBuffer, Experience
from agent import Agent

NUM_STACKS = 4
NUM_MEMORY = 10000

env = Env()
num_action = env.action_space.n

policy_net = DQN(NUM_STACKS, num_action)
target_net = DQN(NUM_STACKS, num_action)

target_net.load_state_dict(policy_net.state_dict())

replay_buffer = ReplayBuffer(NUM_MEMORY)
agent = Agent(
    num_action,
    policy_net,
    target_net,
    replay_buffer,
    batch_size=32,
    gamma=0.99,
    lr=0.0001,
)

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        replay_buffer.push(Experience(state, action, reward, next_state, done))

        agent.update()

        total_reward += reward
        state = next_state

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    agent.update_target_net()

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
