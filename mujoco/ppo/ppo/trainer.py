import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from environments.mujoco_env import MujocoEnv
from ppo.network import ActorCritic, ActorCriticConfig

from tqdm import tqdm
import os

import math

def np_log_gaussian_distribution(mean, std, cns_action):
    log_prob = -0.5*math.log(2*math.pi) -np.log(np.clip(std, 1e-8, 1.0)) -0.5*(((cns_action - mean)/std)**2)
    return np.sum(log_prob, axis=-1)

def torch_log_gaussian_distribution(mean, std, cns_action):
    log_prob = -0.5*math.log(2*math.pi) -torch.log(torch.clamp(std, 1e-8, 1.0)) -0.5*(((cns_action - mean)/std)**2) # ** 2なのかは不明
    return torch.sum(log_prob, dim=-1)

class RolloutBuffer(Dataset):
    def __init__(self, n_worker, T, n_obs, n_cns_action):
        self.states = np.zeros((n_worker, T, n_obs), dtype=np.float32)
        self.cns_actions = np.zeros((n_worker, T, n_cns_action), dtype=np.float32) # not squashed
        self.rewards = np.zeros((n_worker, T, ), dtype=np.float32)
        self.is_terminals = np.zeros((n_worker, T, ), dtype=bool)

        self.squashed_log_probs = np.zeros((n_worker, T), dtype=np.float32)
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
        cns_action = self.cns_actions[worker_idx, idx]
        squashed_log_prob = self.squashed_log_probs[worker_idx, idx]
        advantage = self.advantages[worker_idx, idx]
        value_target = self.value_targets[worker_idx, idx]

        state = torch.tensor(state, dtype=torch.float32)
        cns_action = torch.tensor(cns_action, dtype=torch.float32)
        squashed_log_prob = torch.tensor(squashed_log_prob, dtype=torch.float32)
        advantage = torch.tensor(advantage, dtype=torch.float32)
        value_target = torch.tensor(value_target, dtype=torch.float32)

        state = torch.arctan(state) # normalize

        return state, cns_action, squashed_log_prob, advantage, value_target

class Trainer:
    def __init__(self, args):
        self.env_name = args.env_name
        self.n_worker = args.n_worker
        self.T = args.horizon

        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.c_1 = args.c_1
        self.c_2 = args.c_2

        self.epochs = args.epochs
        self.iter =  args.iter
        self.batch = args.batch
        self.lr = args.lr
        self.evaluate_interval = args.evaluate_interval
        self.save_weights_folder = args.save_weights_folder + args.env_name.lower().replace('-', '_') + '/'

        log_folder = args.log_folder
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        from datetime import datetime
        self.log_path = log_folder + datetime.now().strftime("%Y%m%d_%H%M%S") + '.log'
        with open(self.log_path, 'w') as f:
            f.write(f'PPO {self.env_name} \n')

        self.envs = [MujocoEnv(self.env_name, reward_rescale=True) for _ in range(self.n_worker)]
        self.n_cns_action = self.envs[0].action_space
        self.n_obs = self.envs[0].obs_space

        self.buffer = RolloutBuffer(self.n_worker, self.T, self.n_obs, self.n_cns_action)

        config = ActorCriticConfig()
        config.n_cns_action = self.n_cns_action
        config.n_obs = self.n_obs

        self.ac = ActorCritic(config).to(config.device)

        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.lr)

        self.last_info = None
        self.device = config.device

    def train(self):
        patience = 0
        with tqdm(total=self.iter) as pbar:
            for i in range(self.iter+1):
                self.ac.eval()
                self.rollout()
                self.ac.train()
                for epoch in range(self.epochs):
                    self.run_epoch()

                if i % self.evaluate_interval == 0:
                    self.ac.eval()
                    episode_return = self.evaluate(eval_episodes=20, max_episode_steps=8000)

                    with open(self.log_path ,'a') as f:
                        f.write(f'iter: {i:6d} episode_return: {episode_return} patience {patience:3d}\n')

                    if i > 0:
                        if episode_return < max_episode_return:
                            patience += 1
                            if patience >= 10:
                                break
                        else:
                            os.makedirs(self.save_weights_folder + str(i), exist_ok=True)
                            torch.save(self.ac.state_dict(), self.save_weights_folder + str(i) + '/ac.pth')
                            max_episode_return = episode_return
                    else:
                        max_episode_return = episode_return
                        os.makedirs(self.save_weights_folder + str(i), exist_ok=True)
                        torch.save(self.ac.state_dict(), self.save_weights_folder + str(i) + '/ac.pth')
                            

                pbar.set_description(f'iter: {i:6d} episode_return: {episode_return:.2f} patience {patience:3d}')
                pbar.update(1)

    def run_epoch(self):
        dataloader = DataLoader(self.buffer, batch_size=self.batch, shuffle=True)

        for (states, cns_actions, old_squashed_log_probs, advantages, value_targets) in dataloader:
            states = states.to(self.device)
            cns_actions = cns_actions.to(self.device)
            old_squashed_log_probs = old_squashed_log_probs.to(self.device)
            advantages = advantages.to(self.device)
            value_targets = value_targets.to(self.device)

            self.optimizer.zero_grad()

            (means, stds), values = self.ac(states)

            log_probs = torch_log_gaussian_distribution(means, stds, cns_actions)
            # squashed Gaussian log prob
            squashed_log_probs = log_probs - torch.sum(torch.log(torch.clamp(1 - torch.tanh(cns_actions)**2, 1e-8, 1.0)), dim=-1)

            ratios = torch.exp(squashed_log_probs - old_squashed_log_probs)
            
            clip_loss = torch.min(ratios * advantages, torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages)
            vf_loss = F.mse_loss(values, value_targets)
            clip_loss = torch.mean(clip_loss)
            loss = -clip_loss + self.c_1 * vf_loss

            loss.backward()
            self.optimizer.step()

    def rollout(self):
        if self.last_info is None:
            next_states = np.zeros((self.n_worker, self.n_obs), dtype=np.float32)
            for i in range(self.n_worker):
                next_states[i] = self.envs[i].reset()
        else:
            last_states = self.last_info

            next_states = last_states

        for t in range(self.T):
            states = next_states

            next_states = np.zeros((self.n_worker, self.n_obs), dtype=np.float32)
            cns_actions = np.zeros((self.n_worker, self.n_cns_action), dtype=np.float32)
            rewards = np.zeros((self.n_worker,), dtype=np.float32)
            is_terminals = np.zeros((self.n_worker,), dtype=bool)        

            (means, stds), values = self.ac.infer(states, batch=True)
            cns_actions = np.random.normal(means, stds)

            squashed_cns_actions = np.tanh(cns_actions)

            log_probs = np_log_gaussian_distribution(means, stds, cns_actions)

            squashed_log_probs = log_probs - np.sum(np.log(np.clip(1 - np.tanh(cns_actions)**2, 1e-8, 1.0)), axis=-1)

            for i in range(self.n_worker):
                next_state, reward, done, info = self.envs[i].step(squashed_cns_actions[i])
                next_states[i] = next_state
                rewards[i] = reward
                is_terminals[i] = done

                if is_terminals[i]:
                    next_states[i] = self.envs[i].reset()

            self.buffer.states[:,t] = states
            self.buffer.cns_actions[:,t] = cns_actions
            self.buffer.rewards[:,t] = rewards
            self.buffer.is_terminals[:,t] = is_terminals
            self.buffer.squashed_log_probs[:,t] = squashed_log_probs
            self.buffer.values[:,t] = values

        last_states = next_states
        self.last_info = last_states

        self.calculate_advantages_and_value_targets()

    def calculate_advantages_and_value_targets(self):
        advantages = np.zeros((self.n_worker, ), dtype=np.float32)
        last_states = self.last_info
        next_states = last_states

        _, next_values = self.ac.infer(next_states, batch=True)

        for t in reversed(range(self.T)):
            rewards = self.buffer.rewards[:,t]
            is_terminals = self.buffer.is_terminals[:,t]
            values = self.buffer.values[:,t]

            deltas = rewards + self.gamma * (1 - is_terminals) * next_values - values
            advantages = deltas + self.gamma * self.lamda * (1 - is_terminals) * advantages
            value_targets = advantages + values

            self.buffer.advantages[:,t] = advantages
            self.buffer.value_targets[:,t] = value_targets

            next_values = values

    def evaluate(self, eval_episodes, max_episode_steps):
        env = MujocoEnv(self.env_name)

        episode_returns = []
        
        for e in range(eval_episodes):
            next_state = env.reset()
            reward = 0.0
            done = False
            cns_action = 0.0

            episode_return = 0.0
            for s in range(max_episode_steps):
                state = next_state
                (mean, std), _ = self.ac.infer(state) # (1,), (1,)
                cns_action = np.random.normal(mean, std)
                squashed_cns_action = np.tanh(cns_action)

                next_state, reward, done, info = env.step(squashed_cns_action)
                episode_return += reward

                if done:
                    break
            episode_returns.append(episode_return)
        return sum(episode_returns) / len(episode_returns)

