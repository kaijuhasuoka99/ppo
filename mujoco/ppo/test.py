from ppo.network import ActorCritic, ActorCriticConfig
from environments.mujoco_env import MujocoEnv
import torch
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', '-e', type=str, default='Ant-v4')
parser.add_argument('--load_weights_folder', type=str, default='./weights/ant_v4/500/')

args = parser.parse_args()
env_name = args.env_name
load_weights_folder = args.load_weights_folder

env = MujocoEnv(env_name, render=True, render_mode='human')
config = ActorCriticConfig()
config.n_cns_action = env.action_space
config.n_obs = env.obs_space

ac = ActorCritic(config).to(config.device)
ac.load_state_dict(torch.load(load_weights_folder + 'ac.pth', weights_only=True))

next_state = env.reset()
episode_return = 0.0
for s in range(8000):
    state = next_state
    (mean, std), _ = ac.infer(state)
    cns_action = np.random.normal(mean, std)
    squashed_cns_action = np.tanh(cns_action)

    next_state, reward, done, info = env.step(squashed_cns_action)
    episode_return += reward

    if done:
        break

print(f'Episode Return: {episode_return}')
