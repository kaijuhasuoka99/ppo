# PPO Atari Implementation

## Description
This repository provides a Proximal Policy Optimization (PPO) algorithm that does not run rollouts in parallel.
The Atari implementation of PPO handles a multi-agent rollout with 8 actors, but as the number of agents increases, neural network computation on the CPU and parallel rollout will run slower on a home desktop.
Therefore, we tried to improve the training speed by batch processing the neural network computation in the rollout, generating actions, and processing them in For Loop when interacting with the environment.

## Install
