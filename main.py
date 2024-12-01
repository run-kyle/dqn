import gym

env = gym.make("ALE/Breakout-v5", render_mode="human")
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
env.close()
