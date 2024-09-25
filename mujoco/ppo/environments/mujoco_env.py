import gym
import numpy as np
import cv2

def render(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('screen', img)
    cv2.waitKey(1)

class MujocoEnv:
    def __init__(
            self, 
            env_name='Ant-v4',
            render=False,
            reward_rescale=False,
            render_mode='rgb_array'
            ):
        if render:
            self.env = gym.make(env_name, render_mode=render_mode)
        else:
            self.env = gym.make(env_name)
        self.render = render
        self.action_space = self.env.action_space.shape[0]
        self.obs_space = self.env.observation_space.shape[0]

        self.reward_rescale = reward_rescale

        self.render_mode = render_mode

    def step(self, action):
        if np.isnan(action).any():
            print('action is nan.')
            exit()
        next_state, reward, truncation, termination, info = self.env.step(action)
        if self.render:
            if self.render_mode == 'rgb_array':
                render(self.env.render())

        if self.reward_rescale:
            reward = self.rescale_reward(reward)

        return next_state, reward, (truncation or termination), info 

    def reset(self):
        next_state, info = self.env.reset()
        if self.render_mode == 'human':
            self.env.render()
        return next_state
    
    @staticmethod
    def rescale_reward(r):
        return np.sign(r) * (np.sqrt(np.abs(r)+1) - 1) + 10e-3 * r