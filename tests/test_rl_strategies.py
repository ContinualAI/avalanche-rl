import pytest
from avalanche.training.strategies.reinforcement_learning import *
import torch
from avalanche.models.simple_mlp import SimpleMLP
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from avalanche.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator


# for testing purposes implement all methods needed to have a working RL Strategy
# TODO: add to available strategies!
class RandomStrategy(RLBaseStrategy):

    def __init__(
            self, model, optimizer, per_experience_steps,
            rollouts_per_step: int, max_steps_per_rollout: int,
            updates_per_step: int, plugins):
        super().__init__(model, optimizer, per_experience_steps,
                         rollouts_per_step=rollouts_per_step,
                         max_steps_per_rollout=max_steps_per_rollout,
                         updates_per_step=updates_per_step, plugins=plugins)

    def sample_rollout_action(self, observations: torch.Tensor):
        return np.asarray([self.environment.action_space.sample()
                           for i in range(self.n_envs)])

    def update(self, rollouts, n_update_steps: int):
        pass


def make_random_strategy(
        per_experience_steps: int, rollouts_per_step: int,
        max_steps_per_rollout: int, updates_per_step: int=1, plugins=[]):
    model = SimpleMLP()
    optim = Adam(model.parameters())
    return RandomStrategy(
        model, optim, per_experience_steps, rollouts_per_step=rollouts_per_step,
        max_steps_per_rollout=max_steps_per_rollout,
        updates_per_step=updates_per_step, plugins=plugins)

@pytest.mark.parametrize('rollouts_per_step, max_steps_per_rollout', [(1, -1), (-1, 1), (10, 10)])
def test_rollouts(rollouts_per_step: int,max_steps_per_rollout: int):
    # make strategy
    test_strategy = make_random_strategy(1, rollouts_per_step, max_steps_per_rollout) 
    # make scenario
    scenario = gym_benchmark_generator(['CartPole-v1'], n_parallel_envs=1)

    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Env ", experience.env)
        strategy.train(experience)
        print('Training completed')