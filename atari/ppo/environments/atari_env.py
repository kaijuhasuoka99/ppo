import gym
import numpy as np
from collections import deque

import cv2
# cv2 1.6 times faster than PIL

def render(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('screen', img)
    cv2.waitKey(1)

class AtariEnv:
    def __init__(
                self, 
                env_name,
                render=False,
                frame_skip=4,
                resize=(84, 84),
                gray_scale=True,
                frame_stack=4,
                reward_clip=False,
                reward_rescale=False,
                life_loss=False,
                noop_max=30
                ):
        self.env = gym.make(env_name)
        self.render = render
        self.action_space = self.env.action_space.n
        self.frame_skip=frame_skip
        self.frame_skip_buffer = [
            np.zeros(self.env.observation_space.shape, dtype=np.uint8),
            np.zeros(self.env.observation_space.shape, dtype=np.uint8)
        ]
        self.resize = resize
        self.gray_scale = gray_scale
        self.color = False if self.gray_scale else True

        self.frame_stack = frame_stack

        assert not(reward_rescale and reward_clip)
        self.reward_rescale = reward_rescale
        self.reward_clip = reward_clip

        self.life_loss = life_loss
        self.lives = 0

        assert noop_max >= 0
        self.noop_max = noop_max

    def step(self, action):
        reward = 0.0
        if self.life_loss:
            life_loss = False

        for t in range(self.frame_skip):
            frame, frame_reward, done, info = self.env.step(action)

            if self.render:
                render(frame)

            reward += frame_reward

            if self.life_loss:
                new_lives = info['lives']
                life_loss = life_loss or new_lives < self.lives
                self.lives = new_lives

            if done:
                break

            if t == self.frame_skip - 2:
                self.frame_skip_buffer[1] = frame
            elif t == self.frame_skip - 1:
                self.frame_skip_buffer[0] = frame

        if self.frame_skip > 1:
            frame = np.maximum(self.frame_skip_buffer[0], self.frame_skip_buffer[1])

        if self.resize is not None:
            frame = cv2.resize(frame, self.resize)
        if self.gray_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.color:
            frame = frame.transpose(2, 0, 1)

        self.frame_stack_buffer.append(frame)

        if self.gray_scale:
            state = np.array(self.frame_stack_buffer, dtype=np.uint8)
        elif self.color:
            state = np.concatenate(self.frame_stack_buffer, axis=0, dtype=np.uint8)

        if self.reward_rescale:
            reward = self.rescale_reward(reward)
        elif self.reward_clip:
            reward = self.clip_reward(reward)

        if self.life_loss:
            return state, reward, (life_loss, done), info

        return state, reward, done, info

    def reset(self):
        frame = self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            frame, _, done, _ = self.env.step(0)
            if done:
                frame = self.env.reset()

        if self.resize is not None:
            frame = cv2.resize(frame, self.resize)
        if self.gray_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.color:
            frame = frame.transpose(2, 0, 1)

        self.frame_stack_buffer = deque(maxlen=self.frame_stack)
        self.frame_stack_buffer.extend([frame] * self.frame_stack)

        if self.gray_scale:
            state = np.array(self.frame_stack_buffer, dtype=np.uint8)
        elif self.color:
            state = np.concatenate(self.frame_stack_buffer, axis=0, dtype=np.uint8)

        return state
    
    @staticmethod
    def clip_reward(r):
        return np.sign(r)
    
    @staticmethod
    def rescale_reward(r):
        return np.sign(r) * (np.sqrt(np.abs(r)+1) - 1) + 10e-3 * r