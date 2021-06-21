import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import gym
from gym import Env
from avalanche.benchmarks.rl_benchmark import RLExperience, RLScenario
from typing import Union, Optional, Sequence, List, Dict
from dataclasses import dataclass
import numpy as np
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies.rl_utils import *
from collections import defaultdict


@dataclass
class Step:
    # holds vectorized environment steps result
    states: Union[np.ndarray, torch.Tensor]
    actions: Union[np.ndarray, torch.Tensor]
    dones: Union[bool, np.ndarray]
    rewards: Union[float, np.ndarray]
    next_states: Union[np.ndarray, torch.Tensor]

    @property
    def n_envs(self):
        # FIXME: return self.states.shape[0] 
        return 1


@dataclass
class Rollout:
    steps: List[Step]

    @property
    def n_envs(self):
        return self.steps[0].n_envs

    # TODO: cache property calls
    @property
    def rewards(self):
        """
            Returns all rewards gathered at each step of this rollout.
        """
        # batch dimension over number of parallel environments, instatiate transpose for faster assignment
        rwds = torch.zeros(
            (len(self.steps), self.n_envs),
            dtype=torch.float32)
        for i, step in enumerate(self.steps):
            rwds[i] = step.rewards
        return rwds.T

    @property
    def dones(self):
        """
            Returns terminal state flag of each step of this rollout.
        """
        # batch dimension over number of parallel environments, instatiate transpose for faster assignment
        dones = torch.zeros(
            (len(self.steps), self.n_envs),
            dtype=torch.float32)
        for i, step in enumerate(self.steps):
            dones[i] = step.dones
        return dones.bool().T

    @property
    def actions(self):
        """
            Returns all actions taken at each step of this rollout.
        """
        # batch dimension over number of parallel environments, instatiate transpose for faster assignment
        actions = torch.zeros(
            (len(self.steps), self.n_envs),
            dtype=torch.float32)
        for i, step in enumerate(self.steps):
            actions[i] = step.actions
        return actions.T

    def get_obs(self, prop_name: str, as_tensor=True):
        """
            Returns all observations gathered at each step of this rollout.
        """
        mlib = torch if as_tensor else np
        obs = mlib.zeros(
            (len(self.steps), *self.steps[0].states.shape),
            dtype=mlib.float32)
        for i, step in enumerate(self.steps):
            obs[i] = getattr(step, prop_name)

        # batch dimension over number of parallel environments
        # FIXME: think about batch dim for images return mlib.swapaxes(obs, 0, 1)
        return obs

    @property
    def observations(self):
        return self.get_obs('states')

    @property
    def next_observations(self):
        return self.get_obs('next_states')


# TODO: evaluation. probably we can inherit from base strategy
class RLBaseStrategy:
    def __init__(
                self, model: nn.Module, optimizer: Optimizer, per_experience_episodes: int, per_experience_steps: int = -1,
                rollouts_per_episode: int = 1, max_steps_per_rollout: int = -1, updates_per_step: int = 1,
                device='cpu',
                plugins: Optional[Sequence[StrategyPlugin]] = [],
                # evaluator=default_logger,
                eval_every=-1):
        # self.env = env
        self.model = model
        self.optimizer = optimizer
        self.per_experience_episodes = per_experience_episodes
        self.per_experience_steps = per_experience_steps
        self.rollouts_per_episode = rollouts_per_episode
        self.max_steps_per_rollout = max_steps_per_rollout
        self.updates_per_step = updates_per_step
        self.total_steps = 0
        self.episode_no = 0
        self._obs = None
        self.plugins = plugins
        self.device = device

    def before_rollout(self, **kwargs):
        for p in self.plugins:
            p.before_rollout(self, **kwargs)

    def after_rollout(self, **kwargs):
        for p in self.plugins:
            p.after_rollout(self, **kwargs)

    def sample_rollout_action(self, observations: torch.Tensor):
        raise NotImplementedError(
            "`sample_rollout_action` must be implemented by every RL strategy")

    def rollout(
            self, env: Env, n_rollouts: int, max_steps: int = -1) -> List[Rollout]:
        # gather experience from env
        t = 0
        rollouts = []
        for _ in range(n_rollouts):
            step_experiences = []
            # reset environment only after completing episodes
            if self._obs is None:
                self._obs = env.reset()
            done = False
            while not done:
                action = self.sample_rollout_action(self._obs) 
                # TODO: handle automatic reset for parallel envs with different lenght (must concatenate different episodes)
                next_obs, reward, done, _ = env.step(action)
                step_experiences.append(
                    Step(self._obs, action, done, reward, next_obs))
                t += 1
                self._obs = next_obs
                # TODO: quit when one env is done?
                if done:
                    self._obs = None
                    break
                if max_steps > 0 and t >= max_steps:
                    break
            rollouts.append(Rollout(step_experiences))
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
        self.steps = 0
        # either run N episodes or steps depending on specified parameter
        step_mode = self.per_experience_steps > 0
        # TODO: keep only episode threshold, return done in rollouts?
        n = self.per_experience_steps if self.per_experience_steps > 0 else self.per_experience_episodes

        # Data Adaptation (e.g. add new samples/data augmentation)
        # self.before_train_dataset_adaptation(**kwargs)
        # self.train_dataset_adaptation(**kwargs)
        # self.after_train_dataset_adaptation(**kwargs)
        # self.make_train_dataloader(**kwargs)
        self.environment = self.make_env(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        # self.model_adaptation()
        self.make_optimizer()

        for t in range(n):
            rollouts, steps = self.rollout(
                env=self.environment, n_rollouts=self.rollouts_per_episode,
                max_steps=self.max_steps_per_rollout)
            self.steps += steps
            if not step_mode:
                self.episode_no += 1
            self.update(rollouts, self.updates_per_step)
        self.total_steps += self.steps

    def make_optimizer(self):
        # we reset the optimizer's state after each experience.
        # This allows to add new parameters (new heads) and
        # freezing old units during the model's adaptation phase.
        self.optimizer.state = defaultdict(dict)
        self.optimizer.param_groups[0]['params'] = list(self.model.parameters())


from avalanche.models.actor_critic import ActorCriticMLP
from avalanche.models.dqn import ConvDeepQN
from torch.optim import Optimizer
from torch.distributions import Categorical
# inherit from BaseStrategy too?


class A2CStrategy(RLBaseStrategy):

    def __init__(
            self, model: nn.Module, optimizer: Optimizer,
            per_experience_steps: int, max_steps_per_rollout: int = 1,
            value_criterion=nn.MSELoss(),
            discount_factor: float = 0.99, device='cpu',
            plugins: Optional[Sequence[StrategyPlugin]] = [],
            eval_every=-1, policy_loss_weight: float = 0.5,
            value_loss_weight: float = 0.5,):
        super().__init__(
            model, optimizer, per_experience_episodes=-1,
            per_experience_steps=per_experience_steps,
            rollouts_per_episode=1,
            max_steps_per_rollout=max_steps_per_rollout,
            updates_per_step=1, device=device, plugins=plugins,
            eval_every=eval_every)

        # TODO: 'dataloader' calls with pre-processing env wrappers 
        self.model = model
        self.optimizer = optimizer
        self.value_criterion = value_criterion
        self.discount_factor = discount_factor
        self.ac_w = policy_loss_weight
        self.cr_w = value_loss_weight

    def sample_rollout_action(self, observations: torch.Tensor):
        """
            This will process a batch of observations and produce a batch
            of actions to better leverage GPU as in 'batched' A2C.
        Args:
            observations (torch.Tensor): [description]

        Returns:
            [type]: [description]
        """
        # sample action from policy network
        # FIXME: Remove and add vecenv
        observations = observations.unsqueeze(0)
        with torch.no_grad():
            # policy_only forward?
            print("Sampling action!", observations.shape, observations.dtype)
            _, policy_logits = self.model(observations, compute_value=False)
        # FIXME: remove item and add vecenv (alternative np.random.choice(num_outputs, p=np.squeeze(dist)))
        return Categorical(logits=policy_logits).sample().item()

    def update(self, rollouts: List[Rollout], n_update_steps: int):
        # perform gradient step(s) over batch of gathered rollouts
        # TODO: rollout buffer to avoid loop
        for rollout in rollouts:
            # print("Rollout Observation shape", rollout.observations.shape)
            values, policy_logits = self.model(rollout.observations)
            # ~log(softmax(action_logits))
            # print("Rollout Actions shape", rollout.actions.shape)
            log_prob = Categorical(logits=policy_logits).log_prob(rollout.actions)
            # compute next states values
            next_values, _ = self.model(
                rollout.next_observations, compute_policy=False)
            # mask terminal states values
            next_values[rollout.dones] = 0.

            # print("Rollout Rewards shape", rollout.rewards.shape)
            # Actor/Policy Loss Term in A2C: A(s_t, a_t) * grad log (pi(a_t|s_t))
            boostrapped_returns = rollout.rewards + self.discount_factor * next_values
            advantages = boostrapped_returns - values 
            policy_loss = -(advantages * log_prob).mean()

            # Value Loss Term: R_t + gamma * V(S_{t+1}) - V(S_t
            # value_loss = advantages.pow(2)
            value_loss = self.value_criterion(boostrapped_returns, values)

            loss = self.ac_w * policy_loss + self.cr_w * value_loss
            # TODO: accumulate gradients for multi-rollout case
            self.optimizer.zero_grad()
            # TODO: call hooks
            loss.backward()
            self.optimizer.step()


# class DQN(RLBaseStrategy):

#     def __init__(self, model: ConvDeepQN, optimizer: Optimizer,
#                  per_experience_episodes: int, rollouts_per_episode: int):
#         super().__init__(per_experience_episodes, rollouts_per_episode=rollouts_per_episode)
#         self.model = model

#     def sample_rollout_action(self, observation: torch.Tensor):
#         pass

#     def update(self, rollouts: List[Rollout], n_update_steps: int):
#         return super().update(rollouts, n_update_steps)
