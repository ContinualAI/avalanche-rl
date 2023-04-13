import pytest
from avalanche_rl.training.strategies import *
import torch
from avalanche.models.simple_mlp import SimpleMLP
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from avalanche_rl.benchmarks.rl_benchmark_generators import gym_benchmark_generator


# for testing purposes implement all methods needed to have a working RL Strategy
class RandomTestStrategy(RLBaseStrategy):

    def __init__(
            self, model, optimizer, per_experience_steps,
            rollouts_per_step: int, max_steps_per_rollout: int,
            updates_per_step: int, plugins):
        super().__init__(
            model, optimizer, per_experience_steps, criterion=nn.MSELoss(),
            rollouts_per_step=rollouts_per_step,
            max_steps_per_rollout=max_steps_per_rollout,
            updates_per_step=updates_per_step, plugins=plugins)

    def sample_rollout_action(self, observations: torch.Tensor):
        return np.asarray([self.environment.action_space.sample()
                           for i in range(self.n_envs)])

    def update(self, rollouts):
        # simulate loss implementation
        x, y = torch.randn(1, 3, requires_grad=True), torch.randn(1, 3)
        self.loss = torch.sum((x - y)**2) 

        # does nothing but test 
        if self.rollouts_per_step > 0 and self.max_steps_per_rollout <= 0:
            assert len(rollouts) == self.rollouts_per_step
        elif self.rollouts_per_step > 0 and self.max_steps_per_rollout > 0:
            assert len(rollouts) == self.rollouts_per_step
            for roll in rollouts:
                assert len(roll) <= self.max_steps_per_rollout
                # print('roll len', len(roll)) 
        else:
            assert len(rollouts) == 1
            assert len(rollouts[0]) == self.max_steps_per_rollout


def make_random_strategy(
        per_experience_steps: int, rollouts_per_step: int,
        max_steps_per_rollout: int, updates_per_step: int = 1, plugins=[]):
    model = SimpleMLP(input_size=10, num_classes=3)
    optim = Adam(model.parameters())
    return RandomTestStrategy(
        model, optim, per_experience_steps, rollouts_per_step=rollouts_per_step,
        max_steps_per_rollout=max_steps_per_rollout,
        updates_per_step=updates_per_step, plugins=plugins)


@pytest.mark.parametrize('rollouts_per_step, max_steps_per_rollout',
                         [(2, -1),
                          (-1, 10),
                          (10, 10)])
def test_rollouts_single_env(
        rollouts_per_step: int, max_steps_per_rollout: int):
    # make strategy
    test_strategy = make_random_strategy(
        1, rollouts_per_step, max_steps_per_rollout)
    # make scenario
    scenario = gym_benchmark_generator(['CartPole-v1'], n_parallel_envs=1)

    for experience in scenario.train_stream:
        test_strategy.train(experience)


@pytest.mark.parametrize('rollouts_per_step, max_steps_per_rollout',
                         [(1, -1),
                          (-1, 1),
                          (10, 10)])
def test_rollouts_multienv(
        rollouts_per_step: int, max_steps_per_rollout: int):
    # make strategy
    test_strategy = make_random_strategy(
        1, rollouts_per_step, max_steps_per_rollout)
    # make scenario
    scenario = gym_benchmark_generator(['CartPole-v1'], n_parallel_envs=2)

    for experience in scenario.train_stream:
        test_strategy.train(experience)
