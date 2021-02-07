# %%
import gym

# %%
env = gym.make('CartPole-v0')

total_reward = 0.
total_steps = 0
env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()

env.close()
# %%
