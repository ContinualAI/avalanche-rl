import torch
from typing import Union, List
from dataclasses import dataclass, field
import numpy as np
import random


@dataclass
class Step:
    """ Holds vectorized environment steps result of size `n_envs` x D. """
    states: Union[np.ndarray, torch.Tensor]
    actions: Union[np.ndarray, torch.Tensor]
    dones: Union[bool, np.ndarray]
    rewards: Union[float, np.ndarray]
    next_states: Union[np.ndarray, torch.Tensor]
    _post_init: bool = True

    def __post_init__(self):
        if not self._post_init:
            return
        # make sure no graph's ever attached by mistake
        for k in self.__annotations__:
            if k == '_post_init':
                continue
            var = getattr(self, k)
            if type(var) is torch.Tensor:
                var = var.cpu().detach()
                if var.ndim == 1:
                    var = var.view(-1, 1)
            elif type(var) is np.ndarray:
                if var.ndim == 1:
                    var = var.reshape(-1, 1)
            # elif type(var) is not torch.tensor:
                # var = torch.tensor(var)

            setattr(self, k, var)

    @property
    def n_envs(self):
        return self.states.shape[0] 

    def __getitem__(self, actor_idx: int):
        """ Slice an actor step over the n_envs dimension by returning i-th array of each attribute. """
        if actor_idx >= self.n_envs:
            raise IndexError(
                f'indx {actor_idx} is out of bound for axis with size {self.n_envs} (number of parallel envs).')
        return tuple(getattr(self, k)[actor_idx, ...]
                     for k in self.__annotations__ if k != '_post_init')

    def to(self, device: torch.device):
        args = []
        for k in self.__annotations__:
            if k != '_post_init':
                attr = getattr(self, k)
                # we should only deal with arrays or tensors
                if type(attr) is np.ndarray:
                    attr = torch.from_numpy(attr)
                args.append(attr.to(device))

        return Step(*args, _post_init=False)


@dataclass
class Rollout:
    steps: List[Step]
    n_envs: int
    _device: torch.device = torch.device('cpu')
    _unraveled: bool = False
    """ Whether to shuffle steps contained in this rollout. """
    _shuffle: bool = True
    """ Whether to flatten time dimension returning `n_envs`*`len(steps)`xD tensors. """
    _flatten_time: bool = True

    def _pre_compute_unraveled_steps(self):
        """Computes and stores values for `obs`, `rewards`, `dones`, `next_obs` unraveled
           (e.g. of shape `n_env` x `len(steps)` x D) or flattened 
           (e.g. of shape `n_env` * `len(steps)` x D) through time .
           This is only done during the update of the policy network (when needed) 
           to save memory specifically for the case of ReplayMemory.
        """
        if not len(self.steps):
            return False

        for attr in ['states', 'actions', 'rewards', 'dones', 'next_states']:
            attr_shape = getattr(self.steps[0], attr).shape
            attr_type = getattr(self.steps[0], attr).dtype
            attr_type = attr_type if type(
                attr_type) is torch.dtype else getattr(torch, str(attr_type))
            # print("Cache:", attr, attr_shape, attr_type)
            # step dimension first for loop efficiency
            attr_tensor = torch.zeros(
                len(self.steps),
                *attr_shape, dtype=attr_type)
            setattr(self, '_'+attr, attr_tensor)

        # loop through step and add each attribute
        for i, step in enumerate(self.steps):
            for attr in step.__annotations__:
                if attr != '_post_init':
                    # e.g. actions[step_no] = step.actions
                    step_value = getattr(step, attr)
                    # cast to torch
                    sv = torch.from_numpy(step_value) if type(
                        step_value) is np.ndarray else step_value    
                    getattr(self, '_'+attr)[i] = sv

        # swap attr axes to get desidered shape unravelled or flattened shape
        if self.n_envs > 0:

            if self._shuffle and self._flatten_time:
                perm = torch.randperm(
                    attr_tensor.shape[0] * attr_tensor.shape[1])
            elif self._shuffle and not self._flatten_time:
                perm = torch.randperm(self.n_envs)

            for attr in [
                    'states', 'actions', 'rewards', 'dones', 'next_states']:
                attr_tensor = getattr(self, '_'+attr)
                if self._flatten_time:
                    # `n_env` *`len(steps)` x D
                    attr_tensor = attr_tensor.view(
                        attr_tensor.shape[0] * attr_tensor.shape[1],
                        *attr_tensor.shape[2:])
                else:
                    # `n_env` x `len(steps)` x D
                    attr_tensor = torch.transpose(attr_tensor, 1, 0)

                if self._shuffle:    
                    attr_tensor = attr_tensor[perm]

                setattr(self, '_'+attr, attr_tensor)
                # squeeze timestep dimension if a single step is present
                # print(attr, 'tensor shape', attr_tensor.shape, attr_tensor.dtype)
                # if len(self.steps) == 1:
                # TODO: this doesnt really make a difference for networks
                # setattr(self, '_'+attr, attr_tensor.squeeze(0))
                # else:
        return True

    def _get_value(self, attr: str):
        if not len(self.steps):
            return []
        # pre-compute un-raveled steps before accessing one attribute 
        if not self._unraveled:
            self._unraveled = self._pre_compute_unraveled_steps()

        return getattr(self, '_'+attr)

    @property
    def rewards(self):
        """
            Returns all rewards gathered at each step of this rollout.
        """
        return self._get_value('rewards')

    @property
    def dones(self):
        """
            Returns terminal state flag of each step of this rollout.
        """
        # return self._get_scalar('dones', 'bool')
        return self._get_value('dones')

    @property
    def actions(self):
        """
            Returns all actions taken at each step of this rollout.
        """
        # return self._get_scalar('actions', 'int64')
        return self._get_value('actions')

    @property
    def observations(self):
        """
            Returns all observations gathered at each step of this rollout.
        """
        return self._get_value('states')

    @property
    def next_observations(self):
        """
            Returns all 'next step' observations gathered at each step of this rollout.
        """
        return self._get_value('next_states')

    def to(self, device: torch.device):
        """
            Should only do this before processing to avoid filling gpu with replay memory.
        """
        if not self._unraveled:
            self._unraveled = self._pre_compute_unraveled_steps()
        for attr in ['states', 'actions', 'rewards', 'dones', 'next_states']:
            attr_tensor = getattr(self, '_'+attr)
            setattr(self, '_'+attr, attr_tensor.to(device))
        return self

    def __len__(self):
        return len(self.steps)


@dataclass
class ReplayMemory:
    # like a Rollout but with time-indipendent Steps 
    """ Max number of Steps contained inside memory. When trying to add a new Step and size is reached, a previous Step is replaced. """
    size: int
    n_envs: int
    _memory: List[Step] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.steps_counter: int = len(self._memory)

    def _unravel_step(self, step: Step):
        """
            Slice through provided step on `n_envs` dimension returning `n_envs`
            separated Steps. 
        """
        # support even envs not using VectorizedEnv interface
        if self.n_envs < 0:
            yield step
        else:
            for actor_no in range(step.n_envs):
                # avoid adding n_env dimension
                yield Step(*step[actor_no], _post_init=False)

    def add_rollouts(self, rollouts: List[Rollout]):
        """
            Adds a list of rollouts to the memory disentangling steps coming from
            parallel environments/actors so that we can sample more efficiently from 
            a simple list of steps and handle replacing of new experience.

        Args:
            rollouts (List[Rollout]): [description]
        """
        # increase sample counter counting data from different actors as separate samples
        # FIXME: should be more efficient
        for rollout in rollouts:
            for step in rollout.steps:
                for actor_step in self._unravel_step(step):
                    if self.steps_counter < self.size:
                        self._memory.append(actor_step)
                    else:
                        self._memory[self.steps_counter %
                                     self.size] = actor_step
                    self.steps_counter += 1

    def sample_batch(self, batch_dim: int, device: torch.device) -> Rollout:
        """
            Sample a batch of random Steps, returned as a Rollout with time independent Steps
            and no `n_envs` dimension (therefore of shape `batch_dim` x D). 
            Batch is also moved to device just before processing so that we don't risk
            filling GPU with replay memory samples.

        Args:
            batch_dim (int): [description]

        Returns:
            [type]: [description]
        """
        if batch_dim > len(self._memory):
            raise ValueError("Sample dimension exceeds current memory size")
        # faster than alternatives
        return Rollout([self._memory[i] for i in np.random.choice(len(self), size=batch_dim, replace=False)], n_envs=-1).to(device)
        # return Rollout(np.random.choice(self._memory, size=batch_dim, replace=False).tolist(), n_envs=-1).to(device)

    def reset(self):
        self._memory = []
        self.steps_counter = 0

    def __len__(self):
        return len(self._memory)
