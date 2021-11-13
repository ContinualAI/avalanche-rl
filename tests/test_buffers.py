import pytest
from avalanche_rl.training.strategies.buffers import ReplayMemory, Step, Rollout
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


@pytest.mark.parametrize(['type_', 'device'],
                         product(['torch', 'numpy'],
                                 ['cpu', 'cuda:0']))
def test_step(type_, device):
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip("Need a GPU to run this test")

    step = make_step(type_)
    assert step.states.shape == step.next_states.shape
    if type_ == 'torch':
        assert step.dones.dtype == torch.bool
    else:
        assert step.dones.dtype == np.bool

    device_step = step.to(device)
    for attr in device_step.__annotations__:
        if attr != '_post_init':
            assert type(getattr(device_step, attr)) is torch.Tensor
            assert getattr(
                step, attr).shape == getattr(
                device_step, attr).shape
            assert getattr(device_step, attr).device == torch.device(device)


@pytest.mark.parametrize(('type_', 'n_envs', 'flatten'),
                         product(['torch', 'numpy'],
                                 [1, 3, -1], [True, False]))
def test_rollout(type_: str, n_envs: int, flatten):
    # make a rollout
    rollout = Rollout([make_step(type_, n_envs)
                      for _ in range(20)], n_envs=n_envs, _flatten_time=flatten)
    # assert no unravelling was done at first
    assert rollout._unraveled == False
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
            if flatten:
                assert val.shape[0] == n_envs*len(rollout) 
                assert val.shape[1:] == step_val.shape[1:] 
            else:
                assert val.shape[0] == n_envs
                assert val.shape[1] == len(
                    rollout) and val.shape[2:] == step_val.shape[1:] 
        else:
            assert val.shape[0] == len(
                rollout) and val.shape[1:] == step_val.shape 


def test_rollout_slicing():
    steps = [make_step('torch', 1) for _ in range(20)]
    rollout = Rollout(steps, n_envs=1, _shuffle=False)
    # without unravel
    assert not rollout._unraveled and not rollout[0]._unraveled
    assert rollout[1].steps == steps[1]
    assert rollout[:10].steps == steps[:10]
    assert rollout[-10:].steps == steps[-10:]
    # NOTE: this won't guarantee `rollout[:10].observations = rollout.observations[:10]`
    # as `rollout` will get shuffled upon unravelling (that's why we set `_shuffle=False` above)
    obs = rollout[:10].observations
    assert (obs == rollout.observations[:10]).all()
    # with unravel
    actions = rollout.actions
    states = rollout.observations
    assert rollout._unraveled and rollout[0]._unraveled
    assert (rollout[-10:].actions == actions[-10:]
            ).all() and np.may_share_memory(rollout[-10:].actions, actions[-10:])
    assert (rollout[2:4].observations == states[2:4]).all(
    ) and np.may_share_memory(rollout[2:4].observations, states[2:4])


def test_replay_memory():
    n_envs, type_ = 1, 'torch'
    # make some rollouts
    rollouts = [
        Rollout(
            [make_step(type_, n_envs) for _ in range(20)],
            n_envs=n_envs, _flatten_time=True) for _ in range(3)]

    mem = ReplayMemory(50, n_envs)
    mem.add_rollouts(rollouts)
    assert len(mem) == 50 == mem.actual_size
    # check inside tensors
    for attr in ['observations', 'actions', 'rewards', 'dones',
                 'next_observations']:
        attr_tensor = getattr(mem, attr)
        # newest addition on head
        assert (attr_tensor[:20] == getattr(rollouts[-1], attr)).all()
        assert (attr_tensor[20:40] == getattr(rollouts[-2], attr)).all()
        # TODO: we should flip tensor before storing it in order to keep last X steps instead of first X
        assert (attr_tensor[-10:] == getattr(rollouts[-3], attr)[:10]).all()

    # add a rollout with size greater than memory
    # mem = ReplayMemory(50, n_envs)
    rollout = Rollout([make_step(type_, n_envs)
                      for _ in range(100)], n_envs=n_envs, _flatten_time=True, _shuffle=False)
    mem.add_rollouts([rollout])
    assert mem.actual_size == 50
    for attr in ['observations', 'actions', 'rewards', 'dones',
                 'next_observations']:
        attr_tensor = getattr(mem, attr)
        # NOTE: shuffle must be False to assert this
        assert (attr_tensor == getattr(rollout, attr)[-mem.size:]).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_replay_mem_sampling(device):
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip("Need a GPU to run this test")
    mem = ReplayMemory(50, 1)

    with pytest.raises(Exception):
        mem.sample_batch(1, None)
    rollout = Rollout([make_step('torch', 1)
                      for _ in range(10)], n_envs=1, _flatten_time=True)
    mem.add_rollouts([rollout])
    batch = mem.sample_batch(10, device)
    # only move when sampling
    for attr in ['observations', 'actions', 'rewards', 'dones',
                 'next_observations']:
        assert getattr(mem, attr).device == torch.device('cpu')
        assert len(getattr(batch, attr)) == 10
        assert getattr(batch, attr).device == torch.device(device)
    # TODO: this only works if we copy over steps too
    # assert len(batch) == 10
