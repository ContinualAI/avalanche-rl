from avalanche.benchmarks.rl_benchmark import *
import gym
from gym import envs
from typing import *
import numpy as np

def get_all_envs_id():
    all_envs = envs.registry.all()
    return [env_spec.id for env_spec in all_envs]


def gym_benchmark_generator(
        env_names: List[str] = [],
        environments: List[gym.Env] = [],
        n_envs: int = None,
        env_kwargs: Dict[Dict[Any]] = {},
        per_experience_episodes:Union[int, List[int]]=1,
        n_experiences=None, envs_ids_to_sample_from:List[str]=None, *args, **kwargs) -> RLScenario:
    # three ways to create environments from gym
    if env_names is not None and len(env_names):
        envs_ = [gym.make(ename, **env_kwargs.get(ename, {}))
                for ename in env_names]
    elif environments is not None and len(environments):
        envs_ = environments
    elif n_envs is not None and n_envs>0:
        # choose `n_envs` random envs either from the registered ones or from the provided ones
        to_choose = get_all_envs_id() if envs_ids_to_sample_from is not None else envs_ids_to_sample_from
        ids = np.random.choice(to_choose, n_envs)  
        envs_ = [gym.make(ename, **env_kwargs.get(ename, {})) for ename in ids]
    else:
        raise ValueError('')  

    # envs will be cycled through if n_experiences > len(envs_) otherwise only the first n will be used
    n_experiences = n_experiences if n_experiences is not None else len(envs_)         
    # per_experience_episodes variable number of episodes depending on env
    if type(per_experience_episodes) is list:
        assert len(per_experience_episodes) == n_experiences

    return RLScenario(envs=envs_, n_experiences=n_experiences,per_experience_episodes=per_experience_episodes,*args, **kwargs)


def atari_benchmark_generator() -> RLScenario:
    pass
