import torch
from torch import nn
from torch.nn import functional as F

from networks.actor_critic import ActorCritic, ActorCriticConfig
from environments.atari_env import AtariEnv

import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', '-e', type=str, default='BreakoutNoFrameskip-v4')
parser.add_argument('--load_weights_folder', type=str, default='./weights/breakoutnoframeskip_v4/2500/')
parser.add_argument('--max_episode_steps', type=int, default=80000)

args = parser.parse_args()

env_name = args.env_name
load_weights_folder = args.load_weights_folder + '/' if not args.load_weights_folder.endswith('/') else args.load_weights_folder

max_episode_steps = args.max_episode_steps

config = ActorCriticConfig()
embed_dim = config.embed_dim
device = config.device
env = AtariEnv(env_name, frame_skip=4, frame_stack=4, render=True)
config.n_action = env.action_space
ac = ActorCritic(config).to(device)
ac.load_state_dict(torch.load(load_weights_folder + 'ac.pth', weights_only=True))

# play
next_state = env.reset()
episode_return = 0.0
for s in range(max_episode_steps):
    state = next_state
    p, _ = ac.infer(state)
    action = np.random.choice(len(p), p=p)

    next_state, reward, done, info = env.step(action)
    episode_return += reward

    if done:
        break

print(f'Episode Return: {episode_return}')
