from gym.wrappers.atari_preprocessing import AtariPreprocessing
import pytest
import gym
from avalanche_rl.training.strategies.vectorized_env import VectorizedEnvironment
import numpy as np
import ray
from avalanche_rl.training.strategies.env_wrappers import Array2Tensor, FrameStackingWrapper, RGB2GrayWrapper, CropObservationWrapper
import torch
from gym import Env
import gym.spaces
from avalanche_rl.training.strategies.buffers import Step, Rollout


class CustomTestEnv(Env):

    def __init__(self):
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(10)

    def step(self, action):
        return np.float32(float(action)), action, 0., {'action': action}

    def reset(self):
        return np.float32(1.)


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
    while True:
        actions = np.asarray([env.action_space.sample() for _ in range(4)])
        obs, _, done, info = env.step(actions)
        # get envs which are done
        dones_idx = done.reshape(-1, 1).nonzero()[0]
        assert len(info) == 4

        for idx in range(4):
            if idx in dones_idx:
                assert 'terminal_observation' in info[idx]
                assert info[idx]['terminal_observation'].shape == obs[idx].shape
                # terminal obs different from initial one
                assert sum(info[idx]['terminal_observation'] - obs[idx]) != 0
            else:
                assert 'terminal_observation' not in info[idx]

        if done.any():
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

    # wrap each actor environment to reduce array shape before gathering results 
    venv = VectorizedEnvironment(env, 2, auto_reset=False, wrappers_generators=[
                                 RGB2GrayWrapper, CropObservationWrapper])
    # can be wrapped like any env as long as extra `n_envs` dimension is taken into consideration 
    venv = Array2Tensor(venv)

    vobs = venv.reset()
    # check environments are independent
    with pytest.raises(Exception):
        # this should require env to be reset
        obs = env.step(0)

    obs = env.reset()
    assert vobs.ndim == (obs.ndim + 1)
    assert vobs.shape[1:] == obs.shape
    done = False
    while not done:
        actions = np.asarray([venv.action_space.sample() for _ in range(2)])
        vobs, r, dones, _ = venv.step(actions)
        obs, r, done, _ = env.step(actions[0])
        assert vobs.shape[1:] == obs.shape
        assert type(vobs) is torch.Tensor


def test_env_different():
    try:
        env = gym.make('PongDeterministic-v4')
    except Exception as e:
        pytest.skip(
            "Install `pip install gym[atari]` to run this test + download atari ROMs as explained here https://github.com/openai/atari-py#roms.")
    env = VectorizedEnvironment(env, 3, auto_reset=True)
    # init observation is always the same
    env.reset()
    for _ in range(10):
        # pong actions ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        actions = np.array([0, 2, 3], dtype=np.int32).reshape(3, 1)
        obs, _, done, info = env.step(actions)
        for i in range(obs.shape[0]-1):
            for j in range(i+1, obs.shape[0]):
                assert np.abs(np.sum(obs[i]-obs[j])) > 0

    env.close()


def test_atari_wrapped():
    try:
        env = gym.make('PongNoFrameskip-v4')
    except Exception as e:
        pytest.skip(
            "Install `pip install gym[atari]` to run this test + download atari ROMs as explained here https://github.com/openai/atari-py#roms.")
    wrappers = [AtariPreprocessing, FrameStackingWrapper]
    env = VectorizedEnvironment(
        env, 3, auto_reset=True, wrappers_generators=wrappers)
    # this must be done outside as venv uses numpy as sole interface
    env = Array2Tensor(env)
    # check environments are independent
    env.actors[0].reset.remote()
    with pytest.raises(Exception):
        # this should require env to be reset
        env.actors[1].reset.step(0)
    with pytest.raises(Exception):
        env.actors[2].reset.step(0)

    # init observation is always the same
    obs = env.reset()
    assert obs.shape == (
        3, 4, 84, 84) and isinstance(obs, torch.Tensor)

    # first frame we don't move (?)
    env.step(np.asarray([0, 2, 3]))
    for i in range(10):
        # pong actions ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        actions = np.array([0, 2, 3], dtype=np.int32)
        obs, _, done, info = env.step(actions)
        # check observations are different
        assert torch.sum(obs[0] - obs[1]).item() != 0 and torch.sum(obs[0] -
                                                                    obs[2]).item() != 0 and torch.sum(obs[1] - obs[2]).item() != 0

    env.close()


@pytest.mark.parametrize('shuffle', [True, False])
def test_custom_env_rollout(shuffle: bool):
    n_steps = 10
    n_envs = 7

    env = CustomTestEnv()
    env = VectorizedEnvironment(env, n_envs, auto_reset=True)

    steps = []
    action = np.arange(n_envs).reshape(-1, 1)
    obs, r, done, _ = env.step(action)
    obs = env.reset()

    for _ in range(n_steps):
        n_obs, r, done, _ = env.step(action)
        step = Step(obs, action, done, r, n_obs)
        steps.append(step)
        obs = n_obs

    rollout = Rollout(steps, n_envs=n_envs, _shuffle=shuffle)
    assert (rollout.next_observations == rollout.actions).all() and (
        rollout.actions == rollout.rewards).all()
    for attr in ['observations', 'actions', 'rewards', 'dones',
                 'next_observations']:
        tensor = getattr(rollout, attr)
        assert tensor.shape == torch.Size([n_steps*n_envs, 1])
