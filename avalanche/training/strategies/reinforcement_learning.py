import torch
import torch.nn as nn
import gym
from gym import Env
from avalanche.benchmarks.rl_benchmark import RLExperience, RLScenario
from typing import Union, Optional, Sequence, List, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class Step:
    state: Union[np.ndarray, torch.Tensor]
    action: Union[np.ndarray, torch.Tensor]
    done: bool
    reward: float
    next_state: Union[np.ndarray, torch.Tensor]

@dataclass
class Rollout:
    steps: List[Step]


# TODO: evaluation
class RLBaseStrategy:
    # TODO: model?
    def __init__(
            self, per_experience_episodes: int,
            rollouts_per_episode: int = 1):
        self.per_experience_episodes = per_experience_episodes
        self.rollouts_per_episode = rollouts_per_episode
        self.total_steps = 0

    def pre_rollout(self):
        pass

    def post_rollout(self):
        pass

    def sample_rollout_action(self, observation: torch.Tensor):
        pass

    def rollout(self, env: Env, n_rollouts: int, max_steps: int=-1, ep_no: int = None)->Rollout:
        # gather experience from env
        t = 0
        rollouts = []
        for r in range(n_rollouts):
            experiences = []
            obs = env.reset()
            done = False
            while not done:
                action = self.sample_rollout_action(obs) 
                next_obs, reward, done, _ = env.step(action)
                experiences.append(Step(obs, action, done, reward, next_obs))
                t+= 1
                if max_steps > 0 and t > max_steps:
                    break
            rollouts.append(Rollout(experiences))
        return rollouts, t

    # NOTE: we can have multiple independent losses here 
    def update(self, rollouts: List[Rollout], n_update_steps: int):
        pass

    # FIXME: support old callbacks
    def before_training(self, **kwargs):
        for p in self.plugins:
            p.before_training(self, **kwargs)

    def after_training(self, **kwargs):
        for p in self.plugins:
            p.after_training(self, **kwargs)

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
            self.train_exp(self.experience, eval_streams, **kwargs)
        self.after_training(**kwargs)

        res = self.evaluator.get_last_metrics()
        return res

    def train_exp(self, experience: RLExperience):
        self.environment = experience.environment
        self.steps = 0
        for ep in self.per_experience_episodes:
            # to be performed with vectorized Env for improved performance
            rollouts, steps = self.rollout(env=self.environment, n_rollouts=self.rollouts_per_episode)
            self.steps += steps

            self.update(rollouts)
        self.total_steps += self.steps


from avalanche.models.actor_critic import ActorCritic
from avalanche.models.dqn import ConvDeepQN
from torch.optim import Optimizer
from torch.distributions import Categorical
# inherit from BaseStrategy too?
class A2CStrategy(RLBaseStrategy):
    # TODO: 'dataloader' calls with pre-processing env wrappers 
    def __init__(self, model: ActorCritic, optimizer: Optimizer, per_experience_episodes: int, rollouts_per_episode: int):
        super().__init__(per_experience_episodes, rollouts_per_episode=rollouts_per_episode)
        self.model = model

    def sample_rollout_action(self, observation: torch.Tensor):
        # sample action from policy network
        with torch.no_grad():
            # policy_only forward?
            value, policy_logits = self.model(observation)
        return Categorical(logits=policy_logits).sample()

    def update(self, rollouts: List[Rollout], n_update_steps: int):
        # perform one gradient over all gathered rollouts
        values, policy_logits = self.model(rollouts.actions)
        # ~log(softmax(action_logits))
        log_prob = Categorical(logits=policy_logits).log_prob()
        # TODO:


class DQN(RLBaseStrategy):

    def __init__(self, model: ConvDeepQN, optimizer: Optimizer,per_experience_episodes: int, rollouts_per_episode: int):
        super().__init__(per_experience_episodes, rollouts_per_episode=rollouts_per_episode)
        self.model = model

    def sample_rollout_action(self, observation: torch.Tensor):
        pass

    def update(self, rollouts: List[Rollout], n_update_steps: int):
        return super().update(rollouts, n_update_steps)

