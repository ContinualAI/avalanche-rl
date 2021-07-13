import ray
import gym
from typing import Callable, List, Union, Dict, Any
import numpy as np
import multiprocessing
import types

# ref https://docs.ray.io/en/master/actors.html#creating-an-actor
# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html


@ray.remote
class Actor:
    # we can implement a3c version by having each actor own a copy of the network
    def __init__(
            self, env: Union[gym.Env, Callable],
            actor_id: int, env_kwargs=dict(),
            auto_reset: bool = True) -> None:
        if isinstance(env, gym.Env):
            self.env = env
        else:
            self.env: gym.Env = env(env_kwargs)

        self.id = actor_id
        # allows you to have batches of fixed size independently of episode termination
        self.auto_reset = auto_reset

    def step(self, action: Union[float, int, np.ndarray]):
        """ Actions are computed in batch by the policy network on main process, 
            then sent to actors either over network (distributed setting) or
            shared memory (local setting) leveraging ray backend.
        Args:
            action (Union[float, int, np.ndarray]): [description]

        Returns:
            [type]: [description]
        """
        # try casting to numpy array
        if type(action) is not np.ndarray:
            action = np.array(action)
        
        next_obs, reward, done, info = self.env.step(action)
        # auto-reset episode: obs you get is actually the first of the new episode, 
        # while the last obs is kept inside info
        if self.auto_reset and done:
            info['terminal_observation'] = next_obs.copy()
            next_obs = self.env.reset()
        return next_obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        """
        Renders the environment.
        """
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed: int = None):
        """
        Sets the seed for this env's random number generator(s).
        """
        return self.env.seed(seed)

    def environment(self):
        # for degub purpose, you shouldn't need to call this method as it copies over the env
        return self.env


class VectorizedEnvironment(object):
    """
    Vectorized Environment inspired by https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html,
    realized using `ray` as backend for multi-agent parallel environment interaction.
    Only supports numpy array to avoid facilitate object passing in distributed setting.

    """

    def __init__(
            self, envs: Union[Callable[[Dict[Any, Any]], gym.Env], List[gym.Env], gym.Env],
            n_envs: int, env_kwargs=dict(), auto_reset: bool = True,
            ray_kwargs={'num_cpus': multiprocessing.cpu_count()}
            ) -> None:
        # Avoid passing over potentially big objects on the net, prefer creating 
        # env locally to each actor
        assert n_envs > 0, "Cannot initialize a VectorizedEnv with a non-positive number of environments"
        ray.init(ignore_reinit_error=True, **ray_kwargs)
        self.n_envs = n_envs
        if isinstance(envs, types.FunctionType):
            # each env will be copied over to shared memory if the object is provided
            self.env = envs(**env_kwargs)
            envs = [envs] * self.n_envs
        elif isinstance(envs, gym.Env):
            self.env = envs
            envs = [envs for _ in range(self.n_envs)]
        elif type(envs) is list:
            assert len(envs) == n_envs
            self.env = envs[0]
            # envs = [lambda _ : envs[i] for i in range(n_envs)]

        # ray actor pool?
        self.actors = [Actor.remote(
                           envs[i],
                           i, env_kwargs, auto_reset=auto_reset)
                       for i in range(n_envs)]

    def _remote_vec_calls(self, fname: str, *args, **kwargs) -> Union[np.ndarray, List[Any]]:
        promises = [getattr(actor, fname).remote(*args, **kwargs)
                    for actor in self.actors]
        return np.asarray(ray.get(promises))  # .reshape(self.n_envs, -1)

    def __getattr__(self, attr: str):
        if hasattr(self.env, attr):
            return getattr(self.env, attr)
        else:
            raise AttributeError(attr)

    def step(self, actions: np.ndarray):
        assert actions.shape[0] == self.n_envs, 'First dimension must be equal to number of envs'
        promises = [actor.step.remote(actions[i])
                    for i, actor in enumerate(self.actors)]
        # here we assume Env step computation is approximately the same for 
        # all envs. If that wasnt' the case, we should use https://docs.ray.io/en/latest/package-ref.html#ray.wait
        step_results = [[] for _ in range(4)]
        for actor_res in ray.get(promises):
            for i in range(len(actor_res)):
                step_results[i].append(actor_res[i])

        actor_steps = list(map(np.asarray, step_results))
        # rewards as float instead of double
        actor_steps[1] = actor_steps[1].astype(np.float32)

        return actor_steps

    def reset(self) -> np.ndarray:
        return self._remote_vec_calls('reset')

    def render(self, mode='human') -> np.ndarray:
        return self._remote_vec_calls('render', mode=mode)

    def seed(self, seed: int):
        promises = [actor.seed.remote(seed) for actor in self.actors]
        return ray.get(promises)

    def close(self):
        promises = [actor.close.remote() for actor in self.actors]
        ray.wait(promises)
        ray.shutdown()
