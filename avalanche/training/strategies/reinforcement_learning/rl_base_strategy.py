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
from avalanche.training import default_rl_logger
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


class RLBaseStrategy(BaseStrategy):
    def __init__(
            self, model: nn.Module, optimizer: Optimizer,
            per_experience_steps: Union[int, Timestep],
            criterion=nn.MSELoss(),
            rollouts_per_step: int = 1, max_steps_per_rollout: int = -1,
            updates_per_step: int = 1, device='cpu',
            plugins: Optional[Sequence[StrategyPlugin]] = [],
            discount_factor: float = 0.99, evaluator=default_rl_logger,
            eval_every=-1, eval_episodes: int = 1):

        super().__init__(model, optimizer, criterion=criterion, device=device,
                         plugins=plugins, eval_every=eval_every, evaluator=evaluator)

        assert rollouts_per_step > 0 or max_steps_per_rollout > 0, "Must specify at least one terminal condition for rollouts!"

        # if a single number is passed, assume it's steps
        if not isinstance(per_experience_steps, Timestep):
            per_experience_steps = Timestep(int(per_experience_steps))

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
        self.eval_episodes = eval_episodes

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
        self.rewards, self.ep_lengths = [], []
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
            # keep track of all rewards for parallel environments
            self.rewards.extend(rewards.reshape(-1, ).tolist())
            self._obs = next_obs

            # Vectorized env auto resets on done by default, check this flag to count episodes
            if n_rollouts > 0:
                # check if any actor has finished an episode or `max_steps` reached
                if dones.any() or (max_steps > 0 and len(step_experiences) >= max_steps):
                    rollouts.append(
                        Rollout(step_experiences, n_envs=self.n_envs))
                    # minimum lenght among parallel simulation
                    self.ep_lengths.append(len(step_experiences))
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

    def make_train_env(self, **kwargs):
        # TODO: can be passed as argument to specify transformations
        # maintain vectorized env interface without parallel overhead if `n_envs` is 1
        if self.n_envs == 1:
            env = VectorizedEnvWrapper(self.environment, auto_reset=True)
        else:
            import multiprocessing
            cpus = min(self.n_envs, multiprocessing.cpu_count())
            env = VectorizedEnvironment(
                self.environment, self.n_envs, auto_reset=True, ray_kwargs={'num_cpus': cpus})
        # NOTE: `info['terminal_observation']`` is NOT converted to tensor 
        return Array2Tensor(env)

    def make_eval_env(self, **kwargs):
        # during evaluation we do not use a vectorized environment
        return Array2Tensor(self.environment)

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
        # for i, exp in enumerate(eval_streams):
            # if isinstance(exp, RLExperience):
            # eval_streams[i] = [exp]

        self.before_training(**kwargs)
        for self.experience in experiences:
            # make sure env is reset on new experience
            self._obs = None
            self.train_exp(self.experience, eval_streams, **kwargs)
        self.after_training(**kwargs)

        self.is_training = False
        res = self.evaluator.get_last_metrics()
        print("metrics", res)
        return res

    def train_exp(self, experience: RLExperience, eval_streams, **kwargs):
        self.environment = experience.environment
        self.n_envs = experience.n_envs
        self.rollout_steps = 0

        # Data Adaptation (e.g. add new samples/data augmentation)
        # self.before_train_dataset_adaptation(**kwargs)
        # self.train_dataset_adaptation(**kwargs)
        # self.after_train_dataset_adaptation(**kwargs)
        # self.make_train_dataloader(**kwargs)
        self.environment = self.make_train_env(**kwargs)

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

            # periodic evaluation
            self._periodic_eval(eval_streams, do_final=False)

        self.total_steps += self.rollout_steps
        self.environment.close()

        # Final evaluation
        do_final = True
        if self.eval_every > 0 and \
                self.timestep % self.eval_every == 0:
            do_final = False
        self._periodic_eval(eval_streams, do_final=do_final)
        self.after_training_exp(**kwargs)

    def _periodic_eval(self, eval_streams, do_final):
        """ Periodic eval controlled by `self.eval_every`. """
        # Since we are switching from train to eval model inside the training
        # loop, we need to save the training state, and restore it after the
        # eval is done.
        _prev_state = (
            self.timestep,
            self.experience,
            self.environment,
            # self.dataloader,
            self.is_training)

        if (self.eval_every == 0 and do_final) or \
           (self.eval_every > 0 and self.timestep % self.eval_every == 0):
            for exp in eval_streams:
                self.eval(exp)

        # restore train-state variables and training mode.
        self.timestep, self.experience, self.environment = _prev_state[:3]
        # self.dataloader = _prev_state[3]
        self.is_training = _prev_state[3]
        self.model.train()

    @torch.no_grad()
    def eval(
            self, exp_list: Union[RLExperience, Sequence[RLExperience]],
            **kwargs):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        self.is_training = False
        self.model.eval()

        if isinstance(exp_list, RLExperience):
            exp_list: List[RLExperience] = [exp_list]

        self.before_eval(*kwargs)
        for self.experience in exp_list:
            self.environment = self.experience.environment
            # only single env supported during evaluation
            self.n_envs = self.experience.n_envs

            self.environment = self.make_eval_env(**kwargs)

            # Model Adaptation (e.g. freeze/add new units)
            # self.model_adaptation()

            self.before_eval_exp(**kwargs)
            self.evaluate_exp(**kwargs)
            self.after_eval_exp(**kwargs)

        self.after_eval(**kwargs)

        res = self.evaluator.get_last_metrics()

        return res

    def evaluate_exp(self):
        self.rewards, self.ep_lengths = [], []

        for _ in range(self.eval_episodes):
            obs = self.environment.reset()
            done = False
            t = 0
            while not done:
                action = self.model.get_action(obs.unsqueeze(0).to(self.device))
                obs, r, done, info = self.environment.step(action.item())
                # TODO: use info
                self.rewards.append(r)
                t += 1
            self.ep_lengths.append(t)
        # needed if env comes from train stream and is thus shared
        self.environment.reset()
        self.environment.close()
        # return rewards, lengths
