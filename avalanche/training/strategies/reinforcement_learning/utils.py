from gym import Wrapper, ObservationWrapper
import numpy as np
import gym
import torch
import cv2
from typing import Tuple, Union, Dict, Any

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

class CropObservationWrapper(ObservationWrapper):
    def __init__(self, env, resize_shape=(84, 84)):
        super(CropObservationWrapper, self).__init__(env)
        self.resize_shape = resize_shape
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=(0 if old_space.dtype == np.uint8 else 0.),
            high=(255 if old_space.dtype == np.uint8 else 1.),
            shape=resize_shape,
            dtype=old_space.dtype,
        )

    def observation(self, obs):
        return cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)



class FrameStackingWrapper(ObservationWrapper):
    """"Wrapper for image stacking"""

    def __init__(self, env, n_steps=4):
        super(FrameStackingWrapper, self).__init__(env)
        self.buffer = None
        old_space = env.observation_space
        self.dtype = old_space.dtype
        self.observation_space = gym.spaces.Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=(n_steps, *old_space.shape),
            dtype=self.dtype,
        )

    def reset(self):
        """reset env"""
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        """convert observation"""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        # return a copy of current buffer
        return self.buffer.copy()

class Array2Tensor(ObservationWrapper):

    def __init__(self, env):
        super(Array2Tensor, self).__init__(env)

    def observation(self, observation):
        """convert observation"""
        t = torch.from_numpy(observation).float()
        return t


class VectorizedEnvWrapper(Wrapper):
    """ 
    Wraps a single environment maintaining the interface of vectorized environment 
    with none of the overhead involved by running parallel environments.  
    """
    def __init__(self, env: gym.Env, auto_reset: bool = True) -> None:
        super().__init__(env)
        self.auto_reset = auto_reset

    def _unsqueeze_obs(self, obs):
        if obs.shape == self.observation_space.shape:
            if type(obs) is np.ndarray:
                obs = obs.reshape(1, *obs.shape)
            else:
                obs = np.asarray([obs])
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, reward, done, info = super().step(action.item())
        info = np.asarray([info])
        reward = np.asarray([reward], dtype=np.float32)
        done = np.asarray([done])

        # terminal observation not reshaped
        if self.auto_reset and done:
            info[0]['terminal_observation'] = obs.copy()
            obs = self.reset()

        obs = self._unsqueeze_obs(obs)
        
        return obs, reward, done, info

    def reset(self) -> Any:
        obs = super().reset()
        return self._unsqueeze_obs(obs)
