from gym import Wrapper, ObservationWrapper
import numpy as np
import gym
import torch
import cv2

# Env wrappers adapted from pytorch lighting bolts

class RGB2GrayWrapper(ObservationWrapper):
    def __init__(self, env):
        super(RGB2GrayWrapper, self).__init__(env)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=old_space.shape[0:2],
            dtype=old_space.dtype,
        )

    def observation(self, obs):
        # single channel float32
        # return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        # single channel uint8 conversion
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)



class BufferWrapper(ObservationWrapper):
    """"Wrapper for image stacking"""

    def __init__(self, env, n_steps=4, resolution=(84, 84), dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.buffer = None
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0.0 if dtype==np.float32 else 0,
            high=1.0 if dtype==np.float32 else 255,
            shape=(n_steps, *resolution),
            dtype=dtype,
        )

    def reset(self):
        """reset env"""
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        """convert observation"""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class Array2Tensor(ObservationWrapper):

    def __init__(self, env):
        super(Array2Tensor, self).__init__(env)

    def observation(self, observation):
        """convert observation"""
        return torch.from_numpy(observation).float()