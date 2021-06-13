from avalanche.benchmarks.rl_benchmark import *
import unittest
import pytest
import gym
import numpy as np

# class TestRLBenchmark(unittest.TestCase):

@pytest.mark.parametrize('n_envs', [1, 2, 3])
def test_simple_benchmark_creation(n_envs):
    envs = [gym.make('CartPole-v1') for _ in range(n_envs)]
    rl_scenario = RLScenario(envs, n_experiences=n_envs, task_labels=True)        
    assert rl_scenario is not None
    assert rl_scenario.n_experiences == n_envs
    tstream = rl_scenario.train_stream
    assert rl_scenario.n_experiences == len(tstream)
    for i, exp in enumerate(tstream):
        assert exp.current_experience == i
        env = exp.environment     
        assert isinstance(env, gym.Env)   
        obs = env.reset()
        assert isinstance(obs, np.ndarray)

