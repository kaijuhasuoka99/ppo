import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from networks.actor_critic import ActorCritic, ActorCriticConfig
from environments.atari_env import AtariEnv

import numpy as np
from tqdm import tqdm

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', '-e', type=str, default='BreakoutNoFrameskip-v4')
parser.add_argument('--save_weights_folder', type=str, default='./weights/')
parser.add_argument('--n_worker', '-N', type=int, default=64)
parser.add_argument('--horizon', '-T', type=int, default=128)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lamda', type=float, default=0.95)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--c_1', type=float, default=1)
parser.add_argument('--c_2', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=2.5e-4)
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--iter', type=int, default=2500)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--evaluate_interval', type=int, default=100)
parser.add_argument('--log_folder', type=str, default='./logs/')
parser.add_argument('--plot_folder', type=str, default='./plots/')

args = parser.parse_args()

n_worker = args.n_worker
T = args.horizon
gamma = args.gamma
# lamda = args.lamda
epsilon = args.epsilon
c_1 = args.c_1
c_2 = args.c_2

lr = args.lr
batch = args.batch
iter = args.iter
epochs = args.epochs
evaluate_interval = args.evaluate_interval
env_name = args.env_name
save_weights_folder = args.save_weights_folder + args.env_name.lower().replace('-', '_') + '/'

log_folder = args.log_folder
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
from datetime import datetime
log_path = log_folder + datetime.now().strftime("%Y%m%d_%H%M%S") + '.log'

with open(log_path, 'w') as f:
    f.write(f'PPO {env_name} \n')

class RolloutBuffer(Dataset):
    def __init__(self):
        self.states = np.zeros((n_worker, T, 4, 84, 84), dtype=np.uint8)
        self.actions = np.zeros((n_worker, T, ), dtype=np.uint8)
        self.rewards = np.zeros((n_worker, T, ), dtype=np.float32)
        self.is_terminals = np.zeros((n_worker, T, ), dtype=bool)

        self.log_probs = np.zeros((n_worker, T), dtype=np.float32)
        self.values = np.zeros((n_worker, T), dtype=np.float32)
        self.advantages = np.zeros((n_worker, T), dtype=np.float32)
        self.value_targets = np.zeros((n_worker, T), dtype=np.float32)

        self.n_worker = n_worker
        self.T = T

    def __len__(self):
        return self.n_worker * self.T

    def __getitem__(self, idx):
        worker_idx = idx // self.T
        idx = idx % self.T

        state = self.states[worker_idx, idx]
        action = self.actions[worker_idx, idx]
        log_prob = self.log_probs[worker_idx, idx]
        advantage = self.advantages[worker_idx, idx]
        value_target = self.value_targets[worker_idx, idx]

        state = torch.tensor(state, dtype=torch.float32) / 255.0
        action = torch.tensor(action, dtype=torch.long)
        log_prob = torch.tensor(log_prob, dtype=torch.float32)
        advantage = torch.tensor(advantage, dtype=torch.float32)
        value_target= torch.tensor(value_target, dtype=torch.float32)

        return state, action, log_prob, advantage, value_target
    
def calculate_advantages_and_value_targets(next_values):
    advantages = np.zeros((n_worker, ), dtype=np.float32)
    
    for t in reversed(range(T)):
        rewards = buffer.rewards[:,t]
        is_terminals = buffer.is_terminals[:,t]
        values = buffer.values[:,t]

        deltas = rewards + gamma * (1 - is_terminals) * next_values - values
        advantages = deltas + gamma * lamda * (1 - is_terminals) * advantages
        value_targets = advantages + values

        buffer.advantages[:,t] = advantages
        buffer.value_targets[:,t] = value_targets

        next_values = values

def rollout(last_info):
    if last_info is None:
        next_states = np.zeros((n_worker, 4, 84, 84), dtype=np.uint8)
        for i in range(n_worker):
            next_states[i] = envs[i].reset()
    else:
        last_states = last_info

        next_states = last_states

    for t in range(T):
        states = next_states
        next_states = np.zeros((n_worker, 4, 84, 84), dtype=np.uint8)
        actions = np.zeros((n_worker,), dtype=np.uint8)
        rewards = np.zeros((n_worker,), dtype=np.float32)
        is_terminals = np.zeros((n_worker,), dtype=bool)

        probs, values = ac.infer(states, batch=True) # (N, ~)
        actions = np.array([np.random.choice(len(probs[i]), p=probs[i]) for i in range(n_worker)]) # (N, )

        log_probs = np.log(np.clip(probs, 1e-10, 1.0)) # pi(a|s_t)
        log_probs = log_probs[np.arange(n_worker), actions]

        for i in range(n_worker):
            next_state, reward, (life_loss, done), info = envs[i].step(actions[i])
            next_states[i] = next_state
            rewards[i] = reward
            is_terminals[i] = life_loss or done

            if done:
                next_states[i] = envs[i].reset()

        buffer.states[:,t] = states
        buffer.actions[:,t] = actions
        buffer.rewards[:,t] = rewards
        buffer.is_terminals[:,t] = is_terminals
        buffer.log_probs[:,t] = log_probs
        buffer.values[:,t] = values

    last_states = next_states
    last_info = (last_states)

    _, next_values = ac.infer(next_states, batch=True)

    calculate_advantages_and_value_targets(next_values)

    return last_info

config = ActorCriticConfig()
embed_dim = config.embed_dim
device = config.device

envs = [AtariEnv(env_name, reward_rescale=True, frame_skip=4, frame_stack=4, life_loss=True) for _ in range(n_worker)]
config.n_action = envs[0].action_space
buffer = RolloutBuffer()
ac = ActorCritic(config).to(device)
optimizer = torch.optim.Adam(ac.parameters(), lr)
last_info = None

alpha_scheduler = lambda i: 1 - (i / iter)
alpha = 1.0

def run_epoch():
    dataloader = DataLoader(buffer, batch_size=batch, shuffle=True)

    for (states, actions, old_log_probs, advantages, value_targets) in dataloader:
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        advantages = advantages.to(device)
        value_targets = value_targets.to(device)

        optimizer.zero_grad()

        probs, values = ac(states)

        log_probs = torch.log(torch.clamp(probs, 1e-10, 1.0))
        entropy = -torch.sum(probs * log_probs, dim=-1)
        log_probs = torch.gather(log_probs, 1, actions.unsqueeze(-1)).squeeze(-1) # pi(a_t|s_t)

        ratios = torch.exp(log_probs - old_log_probs)
    
        clip_loss = torch.min(ratios * advantages, torch.clamp(ratios, 1-epsilon*alpha, 1+epsilon*alpha) * advantages)
        
        clip_loss = clip_loss
        entropy = entropy
        values = values
        value_targets = value_targets              

        vf_loss = F.mse_loss(values, value_targets)
        clip_loss = torch.mean(clip_loss)
        entropy = torch.mean(entropy)
        loss = -clip_loss + c_1 * vf_loss - c_2 * entropy

        loss.backward()
        optimizer.step()

def evaluate(eval_episodes=20, max_episode_steps=80000):
    env = AtariEnv(env_name, frame_skip=4, frame_stack=4)
    
    episode_returns = []
    max_episode_steps = int(max_episode_steps * 2.0)

    for e in range(eval_episodes):
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
        episode_returns.append(episode_return)
        
    return sum(episode_returns) / len(episode_returns)

import matplotlib.pyplot as plt
eval_returns = []
plot_folder = args.plot_folder
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
plot_path = plot_folder + env_name.lower().replace('-', '_') + '.png'

with tqdm(total=iter) as pbar:
    for i in range(iter+1):
        last_info = rollout(last_info)
        for epoch in range(epochs):
            ac.train()
            run_epoch()
        if i % evaluate_interval == 0:
            ac.eval()
            episode_return = evaluate()

            if not os.path.exists(save_weights_folder + str(i)):
                os.makedirs(save_weights_folder + str(i))
            torch.save(ac.state_dict(), save_weights_folder + str(i) + '/ac.pth')

            with open(log_path, 'a') as f:
                f.write(f'iter: {i:6d} episode_return: {episode_return:.2f} alpha: {alpha:.4f}\n')

            eval_returns.append(episode_return)

        alpha = alpha_scheduler(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * alpha

        pbar.set_description(f'iter: {i:6d} episode_return: {episode_return:.2f} alpha: {alpha:.4f}')
        pbar.update(1)

x = np.arange(0, iter+1, evaluate_interval)
plt.plot(x, eval_returns)
plt.title(env_name)
plt.xlabel('Iteration')
plt.ylabel('Episode Return')
plt.savefig(plot_path)
