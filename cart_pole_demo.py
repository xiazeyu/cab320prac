import gymnasium as gym
import tqdm
env = gym.make('CartPole-v1', render_mode="human")

env.reset()

for _ in tqdm.trange(500):
    env.render()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample()) # take a random action
    done = truncated or terminated
    if done:
        env.reset()
env.close()
