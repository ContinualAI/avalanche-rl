import pytest
from avalanche.training.strategies.reinforcement_learning.reinforcement_learning import ReplayMemory, Step, Rollout
import numpy as np
import torch
from itertools import product


def unsqueeze(a, n=1):
    a = [a for _ in range(n)]
    if type(a[0]) is np.ndarray:
        # return a[np.newaxis, ...]
        return np.stack(a)
    return torch.stack(a)


def make_step(type_: str = 'numpy', n_envs=-1) -> Step:
    if type_ == 'numpy':
        state = np.random.randn(4, 4)
        action = np.random.randint(0, 10, size=1, dtype=np.int32)
        next_state = np.random.randn(4, 4)
        done = np.random.randint(0, 2, size=1).astype(np.bool)
        reward = np.random.rand(1)
    elif type_ == 'torch':
        state = torch.randn(4, 4)
        action = torch.randint(0, 10, size=(1,), dtype=torch.int32)
        next_state = torch.randn(4, 4)
        done = torch.randint(0, 2, size=(1,)).bool()
        reward = torch.rand(1)
    if n_envs > 0:
        return Step(
            unsqueeze(state, n_envs),
            unsqueeze(action, n_envs),
            unsqueeze(done, n_envs),
            unsqueeze(reward, n_envs),
            unsqueeze(next_state, n_envs))
    else:
        return Step(state, action, done, reward, next_state)


@pytest.mark.parametrize('type_', ['torch', 'numpy'])
def test_step(type_):
    step = make_step(type_)
    assert step.states.shape == step.next_states.shape
    if type_ == 'torch':
        assert step.dones.dtype == torch.bool
    else:
        assert step.dones.dtype == np.bool
    if torch.cuda.is_available():
        gpu_step = step.to('cuda:0')
        for attr in gpu_step.__annotations__:
            if attr != '_post_init':
                assert type(getattr(gpu_step, attr)) is torch.Tensor
                assert getattr(
                    step, attr).shape == getattr(
                    gpu_step, attr).shape
                assert getattr(gpu_step, attr).device == torch.device('cuda:0')
    else:
        pytest.skip("Need a GPU to run this test")


@pytest.mark.parametrize(('type_', 'n_envs'),
                         product(['torch', 'numpy'],
                                 [1, 3, -1]))
def test_rollout(type_: str, n_envs: int):
    # make a rollout
    rollout = Rollout([make_step(type_, n_envs)
                      for _ in range(20)], n_envs=n_envs)
    # assert no unravelling was done at first
    for attr in ['states', 'actions', 'rewards', 'next_states']:
        assert getattr(rollout, '_'+attr, None) is None
    # assert unravelling happens after first access
    for i, attr in enumerate(
            ['observations', 'actions', 'rewards', 'next_observations']):
        if i == 0:
            assert not rollout._unraveled
        else:
            assert rollout._unraveled
        val = getattr(rollout, attr, None)
        if 'observation' in attr:
            attr = attr.replace('observation', 'state')
        step_val = getattr(rollout.steps[0], attr)
        assert val is not None and len(val)
        print("val shape", val.shape, step_val.shape)
        if n_envs > 0:
            assert val.shape[0] == n_envs and val.shape[1] == len(
                rollout) and val.shape[2:] == step_val.shape[1:] 
        else:
            assert val.shape[0] == len(
                rollout) and val.shape[1:] == step_val.shape 

    # test to device
    rollout = Rollout([make_step(type_) for _ in range(20)], n_envs=n_envs)
    if torch.cuda.is_available():
        rolloutgpu = rollout.to(torch.device('cuda:0'))
        for attr in ['states', 'actions', 'rewards', 'dones', 'next_states']:
            attr = '_' + attr
            assert isinstance(getattr(rolloutgpu, attr), torch.Tensor)
            assert getattr(
                rollout, attr).shape == getattr(
                rolloutgpu, attr).shape
            assert getattr(
                rolloutgpu, attr).device == torch.device('cuda:0')
    else:
        pytest.skip("Need a GPU to run this test")


@pytest.mark.parametrize(('type_', 'n_envs'),
                         product(['torch', 'numpy'],
                                 [1, 3, -1]))
def test_replay_memory(type_, n_envs):
    # make some rollouts
    rollouts = [Rollout([make_step(type_, n_envs)
                         for _ in range(20)], n_envs=n_envs) for _ in range(3)]

    mem = ReplayMemory(50, n_envs)
    mem.add_rollouts(rollouts)
    assert len(mem) == 50
    # actual number of inserted steps
    assert mem.steps_counter == 60 * abs(n_envs) 
    with pytest.raises(Exception):
        mem.sample_batch(int(1e6), None)

    batch = mem.sample_batch(10, 'cpu')
    assert len(batch) == 10
    assert isinstance(batch._rewards, torch.Tensor)
    assert batch.dones.dtype == torch.bool

    # check unravelling was done
    for attr in ['states', 'actions', 'rewards', 'dones', 'next_states']:
        assert isinstance(getattr(batch, '_'+attr), torch.Tensor)
        assert getattr(batch, '_'+attr).shape[0] == len(batch.steps) 
        assert getattr(batch, '_'+attr).device == torch.device('cpu')
        if n_envs > 0:
            # here we test that rollout steps are already unraveled through n_envs dimension
            # (e.g. if we get obs from different actors we don't want to see that as single -batched- step but multiple steps)
            assert getattr(
                batch, '_'+attr).shape[1:] == getattr(batch.steps[0], attr).shape
        else:
            assert getattr(
                batch, '_'+attr).shape[1:] == getattr(batch.steps[0], attr).shape
