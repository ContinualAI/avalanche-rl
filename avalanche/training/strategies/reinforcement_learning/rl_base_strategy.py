import torch
import torch.nn as nn
from avalanche.training.strategies.base_strategy import BaseStrategy
from torch.optim.optimizer import Optimizer
from gym import Env
from avalanche.benchmarks.rl_benchmark import RLExperience, RLScenario
from typing import Union, Optional, Sequence, List, Tuple
from dataclasses import dataclass
import numpy as np
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies.reinforcement_learning.utils import *
import enum
from avalanche.training.strategies.reinforcement_learning.vectorized_env import VectorizedEnvironment
from .buffers import Rollout, Step


class TimestepUnit(enum.IntEnum):
    STEPS = 0
    EPISODES = 1


@dataclass
class Timestep:
    value: int
    unit: TimestepUnit = TimestepUnit.STEPS

# TODO: evaluation


class RLBaseStrategy(BaseStrategy):
    def __init__(
                self, model: nn.Module, optimizer: Optimizer, per_experience_steps: Union[int, Timestep], criterion=nn.MSELoss(),
                rollouts_per_step: int = 1, max_steps_per_rollout: int = -1, updates_per_step: int = 1,
                device='cpu',
                plugins: Optional[Sequence[StrategyPlugin]] = [],
                discount_factor: float = 0.99,
                # evaluator=default_logger,
                eval_every=-1):

        super().__init__(model, optimizer, criterion=criterion, device=device,
                         plugins=plugins, eval_every=eval_every)

        assert rollouts_per_step > 0 or max_steps_per_rollout > 0, "Must specify at least one terminal condition for rollouts!"

        # if a single integer is passed, assume it's steps
        if type(per_experience_steps) is int:
            per_experience_steps = Timestep(per_experience_steps)

        self.per_experience_steps: Timestep = per_experience_steps
        self.rollouts_per_step = rollouts_per_step
        self.max_steps_per_rollout = max_steps_per_rollout
        self.updates_per_step = updates_per_step
        self.total_steps = 0
        self._obs: torch.Tensor = None
        # self.plugins = plugins
        # self.device = device
        self.gamma = discount_factor
        # defined by the experience
        self.n_envs: int = None

    # Additional callback added by RLBaseStrategy
    def before_rollout(self, **kwargs):
        for p in self.plugins:
            p.before_rollout(self, **kwargs)

    def after_rollout(self, **kwargs):
        for p in self.plugins:
            p.after_rollout(self, **kwargs)

    def sample_rollout_action(self, observations: torch.Tensor) -> np.ndarray:
        """Implements the action sampling a~Pi(s) where Pi is the parameterized
           function we're trying to learn.
           Output of this function should be a numpy array to comply with 
           `VectorizedEnvironment` interface.

        Args:
            observations (torch.Tensor): batch of observations/state at current time t.

        Returns:
            np.ndarray: batch of actions to perform during rollout of shape `n_envs` x A.
        """
        raise NotImplementedError(
            "`sample_rollout_action` must be implemented by every RL strategy")

    def rollout(
            self, env: Env, n_rollouts: int, max_steps: int = -1) -> Tuple[List[Rollout], int]:
        """
        Gather experience from Environment leveraging VectorizedEnvironment for parallel interaction and 
        handling auto reset behavior.
        Args:
            env (Env): [description]
            n_rollouts (int): [description]
            max_steps (int, optional): [description]. Defaults to -1.

        Returns:
            Tuple[List[Rollout], int]: A list of rollouts, one per episode if `n_rollouts` is defined, where an episode
            is considered over as soon as one of the actors returns done=True. 
            Otherwise a single rollout will be returned with the number of steps defined by `max_steps`.
            A combination of both `n_rollouts` and `max_steps` will result in returning `n_rollouts` episodes of
            length at most `max_steps`.
            The number of steps performed is also always returned along with the rollouts.
        """
        # gather experience from env
        t = 0
        rollout_counter = 0
        rollouts = []
        step_experiences = []
        # reset environment on first run
        if self._obs is None:
            self._obs = env.reset()

        while True:
            # sample action(s) from policy moving observation to device; actions of shape `n_envs`xA
            action = self.sample_rollout_action(
                self._obs.to(self.device))

            # observations returned are one for each parallel environment  
            next_obs, rewards, dones, info = env.step(action)

            step_experiences.append(
                Step(self._obs, action, dones, rewards, next_obs))
            t += 1
            self._obs = next_obs

            # Vectorized env auto resets on done by default, check this flag to count episodes
            # TODO: terminal_observation flag
            if n_rollouts > 0:
                # check if any actor has finished an episode or `max_steps` reached
                if dones.any() or (max_steps > 0 and len(step_experiences) >= max_steps):
                    rollouts.append(
                        Rollout(step_experiences, n_envs=self.n_envs))
                    step_experiences = []
                    rollout_counter += 1
                    # TODO: if not auto_reset: self._obs = env.reset

            # check terminal condition(s)
            if n_rollouts > 0 and rollout_counter >= n_rollouts:
                break

            if max_steps > 0 and n_rollouts <= 0 and t >= max_steps:
                rollouts.append(Rollout(step_experiences, n_envs=self.n_envs))
                break

        return rollouts, t

    def update(self, rollouts: List[Rollout], n_update_steps: int):
        raise NotImplementedError(
            "`update` must be implemented by every RL strategy")

    # FIXME: support old callbacks
    def before_training(self, **kwargs):
        for p in self.plugins:
            p.before_training(self, **kwargs)

    def after_training(self, **kwargs):
        for p in self.plugins:
            p.after_training(self, **kwargs)

    def make_env(self, **kwargs):
        # TODO: can be passed as argument to specify transformations
        # maintain vectorized env interface without parallel overhead if `n_envs` is 1
        if self.n_envs == 1:
            env = VectorizedEnvWrapper(self.environment, auto_reset=True)
        else:
            env = VectorizedEnvironment(
                self.environment, self.n_envs, auto_reset=True)
        # NOTE: `info['terminal_observation']`` is NOT converted to tensor 
        return Array2Tensor(env)

    def train(self, experiences: Union[RLExperience, Sequence[RLExperience]],
              eval_streams: Optional[Sequence[Union[RLExperience,
                                                    Sequence[
                                                        RLExperience]]]] = None,
              **kwargs):
        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        # Normalize training and eval data.
        if isinstance(experiences, RLExperience):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]
        for i, exp in enumerate(eval_streams):
            if isinstance(exp, RLExperience):
                eval_streams[i] = [exp]

        self.before_training(**kwargs)
        for self.experience in experiences:
            # make sure env is reset on new experience
            self._obs = None
            self.train_exp(self.experience, eval_streams, **kwargs)
        self.after_training(**kwargs)

        self.is_training = False
        # res = self.evaluator.get_last_metrics()
        # return res

    def train_exp(self, experience: RLExperience, eval_streams=None, **kwargs):
        self.environment = experience.environment
        self.n_envs = experience.n_envs
        self.rollout_steps = 0

        # Data Adaptation (e.g. add new samples/data augmentation)
        # self.before_train_dataset_adaptation(**kwargs)
        # self.train_dataset_adaptation(**kwargs)
        # self.after_train_dataset_adaptation(**kwargs)
        # self.make_train_dataloader(**kwargs)
        self.environment = self.make_env(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        # self.model_adaptation()
        self.make_optimizer()

        self.before_training_exp(**kwargs)

        # either run N episodes or steps depending on specified `per_experience_steps`
        for self.timestep in range(self.per_experience_steps.value):
            self.before_rollout(**kwargs)
            self.rollouts, steps = self.rollout(
                env=self.environment, n_rollouts=self.rollouts_per_step,
                max_steps=self.max_steps_per_rollout)
            self.after_rollout(**kwargs)
            # TODO: to keep track in default evaluator and do that in callback
            self.rollout_steps += steps

            # update must instatiate `self.loss`
            self.update(self.rollouts, self.updates_per_step)

            # Backward
            self.optimizer.zero_grad()
            self.before_backward(**kwargs)
            self.loss.backward()
            self.after_backward(**kwargs)

            # Optimization step
            self.before_update(**kwargs)
            self.optimizer.step()
            self.after_update(**kwargs)
        print("Timesteps performed:", self.timestep+1)
        self.total_steps += self.rollout_steps
        # TODO: self.environment.close()