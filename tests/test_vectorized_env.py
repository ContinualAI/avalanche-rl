import pytest
import gym
from avalanche.training.strategies.reinforcement_learning.vectorized_env import VectorizedEnvironment
import numpy as np
import ray

def make_env(kwargs=dict()):
    return gym.make('CartPole-v1', **kwargs)

def test_no_env():
    with pytest.raises(AssertionError):
        env = VectorizedEnvironment(make_env, 0)

def test_with_env_object():
    env = gym.make('CartPole-v1')
    env = VectorizedEnvironment([env], 1, ray_kwargs={'num_cpus': 1})
    assert len(env.actors) == 1
    ref = env.actors[0].environment.remote()
    env = ray.get(ref)
    assert isinstance(env, gym.Env)

def test_single_env_single_cpu_reset():
    env = VectorizedEnvironment(make_env, 1, ray_kwargs={'num_cpus': 1})
    # check env
    ref = env.actors[0].environment.remote()
    e = ray.get(ref)
    assert isinstance(e, gym.Env)

    obs = env.reset()
    print('obs', obs)
    assert isinstance(obs, np.ndarray)
    # 4 is CartPole obs space size
    assert obs.shape == (1, 4)
    # if on mac os, make sure you install pip install pyglet==1.5.11
    obs = env.render(mode='rgb_array')
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (1, 800, 1200, 3)

# @pytest.mark.parametrize('n_envs', [2, 3])
# def test_multiple_envs_single_cpu_reset(n_envs: int):
#     env = VectorizedEnvironment(make_env, n_envs, ray_kwargs={'num_cpus': 1})
#     obs = env.reset()
#     assert isinstance(obs, np.ndarray)
#     # 4 is CartPole obs space size
#     assert obs.shape == (n_envs, 4)

# @pytest.mark.parametrize('n_envs', [1, 2, 3])    
# def test_multiple_envs_reset(n_envs: int):
#     pass

# @pytest.mark.parametrize('n_envs', [1, 2, 3])    
# def test_multiple_envs_loop(n_envs: int):
#     pass

# def test_auto_reset():
#     pass
