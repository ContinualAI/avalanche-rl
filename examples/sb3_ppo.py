from collections import defaultdict
from typing import Generator, Union, Optional, Sequence, List
import enum
from dataclasses import dataclass
import numpy as np
import torch
from gym import Env
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche_rl.benchmarks.rl_benchmark import RLExperience
from avalanche_rl.training.plugins.strategy_plugin import RLStrategyPlugin
from avalanche_rl.training.strategies.env_wrappers import *
from avalanche_rl.training import default_rl_logger
from avalanche_rl.training.strategies import RLBaseStrategy
from avalanche_rl.training.strategies.vectorized_env import VectorizedEnvironment
from itertools import count
from avalanche.training.plugins.clock import Clock
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
import torch

Observation = Any  # specific to environment
Action = Any  # specific to environment
Transition = Tuple[Observation, Action, float, bool, Observation]
Vectorized = List


class _DummyEnv(gym.Env):
    # NOTE: this is a dummy environment to provide to SB3 class constructors since we don't have access to the environment in CL
    def __init__(self, observation_space, action_space) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.observations = [self.observation_space.sample()]

    def reset(self):
        # NOTE: should only be called once during PPO._setup_learn()
        return self.observations.pop()

    def step(self, action):
        raise NotImplementedError


def transpose_image(image: np.ndarray) -> np.ndarray:
    # NOTE: SB3 ppo wraps image environments with a transpose image wrapper, but here we transpose images explicitly
    #       before passing into the network
    if len(image.shape) == 3:
        return torch.transpose(image, (2, 0, 1))
    assert len(image.shape) == 4, image.shape
    return torch.transpose(image, (0, 3, 1, 2))


class TimestepUnit(enum.IntEnum):
    STEPS = 0
    EPISODES = 1


@dataclass
class Timestep:
    value: int
    unit: TimestepUnit = TimestepUnit.STEPS


class SB3PPOStrategy(BaseStrategy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_envs: int,
        training_experience_limit: Timestep,
        eval_experience_limit: Timestep,
        device="cpu",
        plugins: Sequence[RLStrategyPlugin] = [],
        evaluator=default_rl_logger,
        periodic_eval_every: Optional[Timestep] = None,
        rng_seed: int = 0,
        policy: str = "MlpPolicy",
        **algorithm_kwargs,
    ):
        self.observation_space = observation_space
        self.obs_is_image = len(self.observation_space.shape) >= 3
        self.action_space = action_space
        self.n_envs = n_envs
        self.ppo = PPO(
            policy=policy,
            env=DummyVecEnv(
                [
                    lambda: _DummyEnv(observation_space, action_space)
                    for _ in range(self.n_envs)
                ]
            ),
            seed=rng_seed % (2**32),
            verbose=0,
            device=device,
            **algorithm_kwargs,
        )
        self.ppo._setup_learn(np.inf, eval_env=None)

        # TODO needed for interactive logging
        self.current_experience_steps = Timestep(value=1)

        self.steps_since_last_train = 0
        self.last_dones = [False] * self.n_envs
        self.iteration = 0
        self.num_timesteps = 0

        super().__init__(
            model=None,  # TODO pass in SB3 model?
            optimizer=None,  # TODO pass in SB3 optimizer?
            criterion=None,  # TODO pass in SB3
            device=device,
            plugins=plugins,
            eval_every=periodic_eval_every,
            evaluator=evaluator,
        )

        self.training_experience_limit = training_experience_limit
        self.eval_experience_limit = eval_experience_limit
        self.periodic_eval_every = periodic_eval_every

        # TODO: support Clock?
        for i in range(len(self.plugins)):
            if isinstance(self.plugins[i], Clock):
                self.plugins.pop(i)
                break
        self.plugins: Sequence[RLStrategyPlugin] = self.plugins

    def before_rollout(self, **kwargs):
        for p in self.plugins:
            p.before_rollout(self, **kwargs)

    def after_rollout(self, **kwargs):
        for p in self.plugins:
            p.after_rollout(self, **kwargs)

    def choose_actions(self, obs: Vectorized[Observation]) -> Vectorized[Action]:
        """
        TODO make this method a base RLStrategy method, and implement in SB3PPO subclass
        """
        if self.obs_is_image:
            # if obs is an image, sb3 ppo is expecting it to be transposed
            obs = transpose_image(obs)

        if (
            self.is_training
            and self.ppo.use_sde
            and self.ppo.sde_sample_freq > 0
            and self.steps_since_last_train % self.sde_sample_freq == 0
        ):
            # Sample a new noise matrix
            self.ppo.policy.reset_noise(self.n_envs)

        with torch.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(obs, self.ppo.device)
            actions, self.last_values, self.last_log_probs = self.ppo.policy.forward(
                obs_tensor,
                deterministic=False,  # TODO maybe set to `not self.is_training`?
            )
        actions = actions.cpu().numpy()

        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return actions

    def receive_transitions(self, transitions: Vectorized[Transition]) -> None:
        """
        TODO make this method a base RLStrategy method, and implement in SB3PPO subclass
        """
        if not self.is_training:
            return

        self.steps_since_last_train += 1
        self.num_timesteps += len(transitions)

        # convert from list of transitions to list for each part of a transition
        obss, actions, rewards, dones, next_obss = list(
            map(np.array, zip(*transitions))
        )

        if self.obs_is_image:
            # if obs is image then sb3 expects it to be transposed
            obss = transpose_image(obss)
            next_obss = transpose_image(next_obss)

        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        next_obss_t = obs_as_tensor(next_obss, self.ppo.device)
        with torch.no_grad():
            # Compute value for the last timestep
            next_obss_values = self.ppo.policy.predict_values(next_obss_t)

        for idx, done in enumerate(dones):
            # FIXME: how do we get access to truncated data from TimeLimit
            truncated = False
            if done and truncated:
                rewards[idx] += self.ppo.gamma * next_obss_values[idx].item()

        # add to ppo's buffer
        self.ppo.rollout_buffer.add(
            obss,
            actions,
            rewards,
            self.last_dones,
            self.last_values,
            self.last_log_probs,
        )
        self.last_dones = dones

        if self.steps_since_last_train >= self.ppo.n_steps:
            self.ppo.rollout_buffer.compute_returns_and_advantage(
                last_values=next_obss_values, dones=dones
            )

            self.iteration += 1

            self.ppo.logger.record("time/iterations", self.iteration)
            self.ppo.logger.record("time/total_timesteps", self.num_timesteps)
            self.ppo.logger.dump(step=self.num_timesteps)

            self._before_training_iteration()
            self.ppo.train()
            self._after_training_iteration()

            self.ppo.policy.set_training_mode(False)
            self.steps_since_last_train = 0
            self.ppo.rollout_buffer.reset()

    def _rollout(
        self, env: Env, limit: Timestep
    ) -> Generator[Tuple[float, int], None, None]:
        """
        :return: Generator of (cumulative reward, episode length). yielded when an episode finishes
        """
        episode_lengths = [0] * self.n_envs
        episode_rewards = [0] * self.n_envs

        obss = env.reset()

        num_steps_taken = 0
        num_episodes_completed = 0
        while self._rollout_should_continue(
            num_steps_taken, num_episodes_completed, limit
        ):
            actions = self.choose_actions(obss)
            next_obss, rewards, dones, infos = env.step(actions)
            resulting_obss = [
                info["terminal_observation"] if done else next_obs
                for info, done, next_obs in zip(infos, dones, next_obss)
            ]
            self.receive_transitions(
                list(zip(obss, actions, rewards, dones, resulting_obss))
            )

            for i, done in enumerate(dones):
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1
                num_steps_taken += 1
                if done:
                    num_episodes_completed += 1
                    yield (episode_rewards[i], episode_lengths[i])
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
            obss = next_obss

    def _rollout_should_continue(
        self, num_steps_taken: int, num_episodes_completed: int, limit: Timestep
    ) -> bool:
        more_steps_needed = (
            limit.unit == TimestepUnit.STEPS and num_steps_taken < limit.value
        )
        more_episodes_needed = (
            limit.unit == TimestepUnit.EPISODES and num_episodes_completed < limit.value
        )
        return more_steps_needed or more_episodes_needed

    def make_train_env(self, **kwargs):
        # maintain vectorized env interface without parallel overhead if `n_envs` is 1
        if self.n_envs == 1:
            env = VectorizedEnvWrapper(self.environment, auto_reset=True)
        else:
            import multiprocessing

            cpus = min(self.n_envs, multiprocessing.cpu_count())
            env = VectorizedEnvironment(
                self.environment,
                self.n_envs,
                auto_reset=True,
                wrappers_generators=self.experience.scenario._wrappers_generators[
                    self.environment.spec.id
                ],
                ray_kwargs={"num_cpus": cpus},
            )
        return env

    def make_eval_env(self, **kwargs):
        # NOTE: using vectorization in eval
        return self.make_train_env(**kwargs)

    def train(
        self,
        experiences: Union[RLExperience, Sequence[RLExperience]],
        eval_streams: Optional[
            Sequence[Union[RLExperience, Sequence[RLExperience]]]
        ] = None,
        **kwargs,
    ):
        self.is_training = True

        self._before_training(**kwargs)

        # Normalize training and eval data.
        if isinstance(experiences, RLExperience):
            experiences = [experiences]

        for self.experience in experiences:
            self.train_exp(self.experience, eval_streams, **kwargs)

        self._after_training(**kwargs)

        self.is_training = False

        return self.evaluator.get_last_metrics()

    def train_exp(self, experience: RLExperience, eval_streams, **kwargs):
        assert experience.n_envs == self.n_envs
        self.environment = experience.environment
        self.environment = self.make_train_env(**kwargs)

        self.ep_lengths: Dict[int, List[float]] = defaultdict(lambda: list([0]))
        self.rewards = {
            "curr_returns": np.zeros((self.n_envs,), dtype=np.float32),
            "past_returns": [],
        }

        self._before_training_exp(**kwargs)

        self.before_rollout(**kwargs)
        num_episodes = 0
        num_steps = 0
        for episode_reward, episode_length in self._rollout(
            self.environment, self.training_experience_limit
        ):
            self.rewards["past_returns"].append(episode_reward)
            self.ep_lengths[0].append(episode_length)
            num_episodes += 1
            num_steps += episode_length
            if eval_streams is not None and self._should_periodic_eval(
                num_steps, num_episodes, self.periodic_eval_every
            ):
                self._periodic_eval(eval_streams)
                num_episodes = 0
                num_steps = 0

        self.after_rollout(**kwargs)

        self._after_training_exp(**kwargs)

        self.environment.close()

    def _should_periodic_eval(
        self, num_steps: int, num_episodes: int, frequency: Optional[Timestep]
    ) -> bool:
        return frequency is not None and (
            (
                self.periodic_eval_every.unit == TimestepUnit.STEPS
                and num_steps >= self.periodic_eval_every.value
            )
            or (
                self.periodic_eval_every.unit == TimestepUnit.EPISODES
                and num_episodes >= self.periodic_eval_every.value
            )
        )

    def _periodic_eval(self, eval_streams):
        _prev_state = (
            self.experience,
            self.environment,
            self.n_envs,
            self.is_training,
        )

        for exp in eval_streams:
            self.eval(exp)

        (
            self.experience,
            self.environment,
            self.n_envs,
            self.is_training,
        ) = _prev_state

    @torch.no_grad()
    def eval(self, exp_list: Union[RLExperience, Sequence[RLExperience]], **kwargs):
        if isinstance(exp_list, RLExperience):
            exp_list: List[RLExperience] = [exp_list]

        self.is_training = False

        self._before_eval(*kwargs)

        for self.experience in exp_list:
            self.evaluate_exp(self.experience, **kwargs)

        self._after_eval(**kwargs)

        return self.evaluator.get_last_metrics()

    def evaluate_exp(self, experience: RLExperience):
        assert experience.n_envs == self.n_envs
        self.environment = experience.environment
        self.environment = self.make_eval_env()

        self.eval_ep_lengths = {0: []}
        self.eval_rewards = {"past_returns": []}

        self._before_eval_exp()

        for episode_reward, episode_length in self._rollout(
            self.environment, self.eval_experience_limit
        ):
            self.eval_rewards["past_returns"].append(episode_reward)
            self.eval_ep_lengths[0].append(episode_length)

        self._after_eval_exp()

        self.environment.close()


def main():
    from avalanche_rl.benchmarks.generators.rl_benchmark_generators import (
        gym_benchmark_generator,
    )
    import torch

    device = torch.device("cpu")

    num_envs = 1

    scenario = gym_benchmark_generator(
        ["CartPole-v1"],
        n_parallel_envs=num_envs,
        eval_envs=["CartPole-v1"],
        n_experiences=1,
    )

    dummy_env = gym.make("CartPole-v1")

    strategy = SB3PPOStrategy(
        observation_space=dummy_env.observation_space,
        action_space=dummy_env.action_space,
        n_envs=num_envs,
        training_experience_limit=Timestep(10_000, TimestepUnit.STEPS),
        eval_experience_limit=Timestep(10, TimestepUnit.EPISODES),
        periodic_eval_every=Timestep(1000, TimestepUnit.STEPS),
        device=device,
    )

    print("Starting experiment...")
    results = [strategy.eval(scenario.test_stream)]
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Env ", experience.env)
        print("Current Task", experience.task_label, type(experience.task_label))
        strategy.train(experience, scenario.test_stream)
        results.append(strategy.eval(scenario.test_stream))

    print("Training completed")
    print(strategy.eval(scenario.test_stream))


if __name__ == "__main__":
    main()
