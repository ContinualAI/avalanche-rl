import pytest
import gym
from avalanche.training.strategies.reinforcement_learning.vectorized_env import VectorizedEnvironment
import numpy as np
import ray
from avalanche.training.strategies.rl_utils import RGB2GrayWrapper, BufferWrapper, CropObservationWrapper


def make_env(kwargs=dict()):
    return gym.make('CartPole-v1', **kwargs)


def test_no_env():
    with pytest.raises(AssertionError):
        env = VectorizedEnvironment(make_env, 0)


def test_with_env_object():
    env = gym.make('CartPole-v1')
    env = VectorizedEnvironment(env, 1, ray_kwargs={'num_cpus': 1})
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
    # if on mac os, make sure you install pip install pyglet==1.5.11. won't work on servers
    # obs = env.render(mode='rgb_array')
    # assert isinstance(obs, np.ndarray)
    # assert obs.shape[0] == 1 and obs.shape[-1] == 3


@pytest.mark.parametrize(['n_envs', 'n_cpu'], [[2, 3], [1, 2]])
def test_multiple_envs_reset(n_envs: int, n_cpu: int):
    env = VectorizedEnvironment(
        make_env, n_envs, ray_kwargs={'num_cpus': n_cpu})
    obs = env.reset()
    assert obs.shape == (n_envs, 4)
    env.close()


@pytest.mark.parametrize('n_envs', [1, 2, 3])    
def test_multiple_envs_loop(n_envs: int):
    env = VectorizedEnvironment(make_env, n_envs, auto_reset=False)
    obs = env.reset()
    assert obs.shape == (n_envs, 4)
    done = [False]
    while not any(done):
        actions = np.asarray([env.action_space.sample() for _ in range(n_envs)])
        obs, r, done, _ = env.step(actions)
        assert obs.shape == (n_envs, 4)
        assert r.shape == (n_envs, )
        assert done.shape == (n_envs, )

    env.close()


def test_auto_reset():
    env = VectorizedEnvironment(make_env, 4, auto_reset=True)
    env.reset()
    found = False
    while True:
        actions = np.asarray([env.action_space.sample() for _ in range(4)])
        obs, r, done, info = env.step(actions)
        # it's never done
        dones_idx = done.nonzero()[0]
        assert not len(dones_idx)
        assert len(info) == 4
        for idx in range(4):
            if info[idx]['actual_done']:
                assert 'terminal_observation' in info[idx]
                assert info[idx]['terminal_observation'].shape == obs[idx].shape
                # terminal obs different from starting one
                assert sum(info[idx]['terminal_observation'] - obs[idx]) > 0
                found = True
        if found:
            break

    env.close()


def test_env_wrapping():
    try:
        env = gym.make('Pong-v4')
    except Exception as e:
        pytest.skip(
            "Install `pip install gym[atari]` to run this test + download atari ROMs as explained here https://github.com/openai/atari-py#roms.")
    env = RGB2GrayWrapper(env)
    env = CropObservationWrapper(env, resize_shape=(84, 84))
    env = BufferWrapper(env, resolution=(84, 84))
    venv = VectorizedEnvironment(env, 2, auto_reset=False)

    vobs = venv.reset()
    obs = env.reset()
    assert vobs.ndim == (obs.ndim + 1)
    assert vobs.shape[1:] == obs.shape
    done = False
    while not done:
        actions = np.asarray([venv.action_space.sample() for _ in range(2)])
        vobs, r, dones, _ = venv.step(actions)
        obs, r, done, _ = env.step(actions[0])
        assert vobs.shape[1:] == obs.shape
