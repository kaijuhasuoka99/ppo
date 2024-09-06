# PPO Atari Implementation

## Description
This repository provides a Proximal Policy Optimization (PPO) algorithm that does not run rollouts in parallel.
The Atari implementation of PPO handles a multi-agent rollout with 8 actors, but as the number of agents increases, neural network computation on the CPU and parallel rollout will run slower on a home desktop.
Therefore, we tried to improve the training speed by **Batch Processing** the neural network computation in the rollout, generating actions, and processing them in **For Loop** when interacting with the environment.

## Installation
Create a virtual environment and there,
```bash
pip install -r requirements.txt
```
We have tested with python 3.9.13.
We recommend that you create a new virtual environment, as the gym atari environment might not be able to install because of compatibility with other libraries.
