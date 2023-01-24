from gym import Wrapper, ObservationWrapper
import numpy as np
import gym
import torch
import cv2
from typing import Tuple, Union, Dict, Any, List
from gym.spaces.box import Box

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
            low=old_space.low.min(),
            high=old_space.high.max(),
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
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        """convert observation"""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        # return a copy of current buffer o/w will be referencing same object
        return self.buffer.copy()

class Array2Tensor(ObservationWrapper):
    """ Convert observation from numpy array to torch tensors. """

    def __init__(self, env):
        super(Array2Tensor, self).__init__(env)

    def observation(self, observation):
        t = torch.from_numpy(observation).float()
        return t

class FireResetWrapper(gym.Wrapper):
    """
    Adapated from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py.
    Take action on reset for environments that are fixed until firing.

    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        # assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> np.ndarray:
        action_keys = self.env.unwrapped.get_action_meanings()
        if "FIRE" in action_keys and len(action_keys) >= 3:
            self.env.reset(**kwargs)
            obs, _, done, _ = self.env.step(1)
            if done:
                self.env.reset(**kwargs)
            obs, _, done, _ = self.env.step(2)
            if done:
                self.env.reset(**kwargs)
            return obs
        else:
            return super().reset(**kwargs)

class ClipRewardWrapper(gym.RewardWrapper):
    """
        Clips reward to {-1, 0, 1} depending on its sign.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reward(self, reward: float) -> float:
        return np.sign(reward)

class ReducedActionSpaceWrapper(gym.ActionWrapper):

    def __init__(
            self, env: gym.Env,
            action_space_dim: int,
            action_mapping: Dict[int, int] = {1: 2, 2: 3}) -> None:
        """Re-maps action space to specified values. This is particularly useful with 
           atari environments such as Pong having a default action space which is 
           unnecessarily big (only 3 actions have effect).
           It is also useful when learning a single policy for multiple
           envs, in that case you can re-map model output to specific actions depending
           on the game being played directly from the wrapper (e.g. game1 {0, 1}->{7, 8},
           game2 {0, 1}->{2, 3}).

        Args:
            env (gym.Env): The environment to wrap.
            action_space_dim (int): Dimension of the new action space. This wrapper only
            supports `Discrete` action spaces. 
            action_mapping (Dict[int, int], optional): A dict specifying which actions 
            must be re-mapped from {network output -> game actions} 'codes'. 
            Unspecified actions are left as they are. Defaults to {1: 2, 2: 3}, which in Pong
            removes `FIRE` from actions, leaving NO_OP-LEFT-RIGHT if `action_space_dim` 
            is also set to 3.
        """
        assert action_space_dim > 0, 'action space must be strictly positive'
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(action_space_dim)
        self._action_mapping = action_mapping

    def action(self, action):
        # re-map actions
        return self._action_mapping.get(action, action)

class VectorizedEnvWrapper(Wrapper):
    """ 
    Wraps a single environment maintaining the interface of vectorized environment 
    with none of the overhead involved in running parallel environments.  
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