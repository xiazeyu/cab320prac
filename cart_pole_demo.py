import tensorflow as tf
from tensorflow import keras

import gymnasium as gym
import tqdm
import numpy as np
# env = gym.make('CartPole-v1', render_mode="human")

# env.reset()

# for _ in tqdm.trange(500):
#     env.render()
#     obs, reward, terminated, truncated, info = env.step(env.action_space.sample()) # take a random action
#     done = truncated or terminated
#     if done:
#         env.reset()
# env.close()

model = tf.keras.models.load_model('pg_prac.h5')

env = gym.make("CartPole-v1", render_mode=None)

def render_policy_net(model, n_max_steps=200, seed=42):
    frames = []
    env = gym.make("CartPole-v1", render_mode='human')
    if seed is not None:
        np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    for step in tqdm.trange(n_max_steps):
        frames.append(env.render())
        left_proba = model.predict(obs.reshape(1, -1))
        action = int(np.random.rand() > left_proba)
        # print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        if done:
            break
    env.close()
    return frames
    

for iteration in tqdm.trange(10):
    render_policy_net(model, 300, None)

env.close()