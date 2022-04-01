from collections import defaultdict
from typing import Generator, Union, Optional, Sequence, List
import enum
from dataclasses import dataclass
import numpy as np
import torch
from gym import Env
import gym
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche_rl.benchmarks.rl_benchmark import RLExperience
from avalanche_rl.training.plugins.strategy_plugin import RLStrategyPlugin
from avalanche_rl.training.strategies.buffers import Rollout
from avalanche_rl.training.strategies.env_wrappers import *
from avalanche_rl.training import default_rl_logger
from avalanche_rl.training.strategies.rl_base_strategy import RLBaseStrategy, TimestepUnit, Timestep
from avalanche_rl.training.strategies.vectorized_env import VectorizedEnvironment
from itertools import count
from avalanche.training.plugins.clock import Clock
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, get_policy_from_name, _policy_registry
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from gym.spaces import Discrete
from avalanche_rl.training import default_rl_logger
from avalanche_rl.training.plugins.evaluation import RLEvaluationPlugin
from avalanche_rl.evaluation.metrics.reward import moving_window_stat
from avalanche_rl.logging.interactive_logging import TqdmWriteInteractiveLogger

Observation = Any  # specific to environment
Action = Any  # specific to environment
Transition = Tuple[Observation, Action, float, bool, Observation]
Vectorized = List


class _DummyPolicy(BasePolicy):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def to(self, *args, **kwargs):
        return self

    def _predict(self, *args, **kwargs):
        pass
    def forward(self, *args, **kwargs):
        pass


# NOTE: loss computation, optim step and grad clip happens inside sb3; 
# rollouts and model are handled from the outside 
my_logger = RLEvaluationPlugin(moving_window_stat('ep_length', window_size=10, stats=[
                                                        'mean', 'max', 'std']),
                               moving_window_stat('ep_length', window_size=4, stats=[
                                                        'mean', 'std'], mode='eval'),
                               loggers=[TqdmWriteInteractiveLogger(log_every=1000)])


class SB3PPOStrategy(RLBaseStrategy):
    def __init__(self, 
                 model: BasePolicy, 
                 per_experience_steps: Union[int, Timestep, List[Timestep]],
                 #  sb3_policy: str = "MlpPolicy",
                 # TODO: this is fixed to 1 to comply with sb3 buffer, we must handle buffer differently
                 max_steps_per_rollout: int = 1,
                 device='cpu', 
                 eval_every=-1,
                 eval_episodes: int = 1, 
                 sb3_kwargs=dict(), **kwargs):
        super().__init__(model=model, optimizer=None, per_experience_steps=per_experience_steps, criterion=None, rollouts_per_step=-1, max_steps_per_rollout=max_steps_per_rollout,
                         updates_per_step=1, device=device, eval_every=eval_every, eval_episodes=eval_episodes, evaluator=my_logger, **kwargs)
        # TODO: map internal args such as discount factor, optim.. to sb3 ones
        # TODO: model itself might be instatiated in here but it would be less flexible
        # if isinstance(model, ActorCriticPolicy):
        # self._sb3_policy = "MlpPolicy"
        self._sb3_kwargs = sb3_kwargs
        self.ppo: PPO = None

        self.steps_since_last_train = 0
        self.iteration = 0
        self.num_timesteps = 0
        self.obs_is_image: bool = None

    def _before_training_exp(self, **kwargs):
        self.last_dones = [False] * self.environment.num_envs
        self.ppo = PPO(
            policy=_DummyPolicy,
            env=self.environment,
            # seed=rng_seed % (2**32),
            # verbose=0,
            device=self.device,
            n_epochs=1,  # NOTE: equals to a pass on the same data
            _init_setup_model=False,  # NOTE: model is handled in this class
            ** self._sb3_kwargs,
        )
        self._sb3_setup_model(self.ppo)
        self.ppo._setup_learn(np.inf, eval_env=None)
        return super()._before_training_exp(**kwargs)

    def _sb3_setup_model(self, sb3_algorithm: OnPolicyAlgorithm, seed=None) -> None:
        # from sb3 "on_policy_algorithm"
        # this will create a dummy model (so we dont waste time) then swap it with ours
        sb3_algorithm._setup_model()

        # TODO: can we plug in our buffers here instead? 
        # buffer_cls = DictRolloutBuffer if isinstance(
        #     self.current_observation_space, gym.spaces.Dict) else RolloutBuffer

        # sb3_algorithm.rollout_buffer = buffer_cls(
        #     sb3_algorithm.n_steps,
        #     sb3_algorithm.observation_space,
        #     sb3_algorithm.action_space,
        #     device=sb3_algorithm.device,
        #     gamma=sb3_algorithm.gamma,
        #     gae_lambda=sb3_algorithm.gae_lambda,
        #     n_envs=sb3_algorithm.n_envs,
        # )
        sb3_algorithm.policy = self.model

        # TODO: this is ppo specific, we could even call this method and then swap the model

        # model is moved by framework
        # self.policy = self.policy.to(self.device)

    def make_train_env(self, **kwargs):
        self.obs_is_image = len(self.current_observation_space.shape) >= 3
        env = super().make_train_env(**kwargs)
        if self.obs_is_image:
            # if obs is an image, sb3 ppo is expecting it to be transposed
            return TransposeImageWrapper(env)
        return env

    def train_exp(self, experience: RLExperience, eval_streams, **kwargs):
        self.environment = experience.environment
        self.n_envs = experience.n_envs
        self.rollout_steps = 0
        # one per (parallel) environment
        self.ep_lengths: Dict[int, List[float]] = defaultdict(lambda: list([0]))
        # curr episode returns (per actor) - previous episodes returns 
        self.rewards = {'curr_returns': np.zeros(
                            (self.n_envs,),
                            dtype=np.float32),
                        'past_returns': []}

        # Environment creation
        self.environment = self.make_train_env(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        # TODO: reset sb3 optimizer instead?
        # self.make_optimizer()

        self._before_training_exp(**kwargs)

        # either run N episodes or steps depending on specified `per_experience_steps`
        for self.timestep in range(self.current_experience_steps.value):
            self.before_rollout(**kwargs)
            self.rollouts = self.rollout(
                env=self.environment, n_rollouts=self.rollouts_per_step,
                max_steps=self.max_steps_per_rollout)
            self.after_rollout(**kwargs)

            for self.update_step in range(self.updates_per_step):
                # update must instatiate `self.loss`
                self.update(self.rollouts)

            self._after_training_iteration(**kwargs)
            # periodic evaluation
            self._periodic_eval(eval_streams, do_final=False)

        self.total_steps += self.rollout_steps
        self.environment.close()

        # Final evaluation
        self._periodic_eval(eval_streams, do_final=(
            self.timestep % self.eval_every != 0))
        self._after_training_exp(**kwargs)

    def sample_rollout_action(self, observations: torch.Tensor) -> np.ndarray:
        if (
            self.is_training
            and self.ppo.use_sde
            and self.ppo.sde_sample_freq > 0
            and self.steps_since_last_train % self.sde_sample_freq == 0
        ):
            # Sample a new noise matrix
            self.ppo.policy.reset_noise(self.n_envs)
        # print("Obs shape", observations.shape, observations.dtype, observations.device)
        with torch.no_grad():
            # Convert to pytorch tensor or to TensorDict
            # obs_tensor = obs_as_tensor(obs, self.ppo.device)
            actions, self.last_values, self.last_log_probs = self.ppo.policy.forward(
                observations,
                deterministic=not self.is_training,
            )
        # print("Sampled Actions shape", actions.shape, actions.dtype)
        actions = actions.cpu().numpy()

        # Clip the actions to avoid out of bound error
        if isinstance(self.current_action_space, gym.spaces.Box):
            actions = np.clip(actions, self.current_action_space.low,
                              self.current_action_space.high)

        return actions

    def update(self, rollouts: List[Rollout]):
        if not self.is_training:
            return

        for rollout in rollouts:
            self.steps_since_last_train += 1
            self.num_timesteps += len(rollout)

            # move samples to device for processing and expect tensor of shape `timesteps`x`n_envs`xD`
            rollout = rollout.to(self.device)

            # # convert from list of transitions to list for each part of a transition
            # obss, actions, rewards, dones, next_obss = list(
            #     map(np.array, zip(*transitions))
            # )

            # if isinstance(self.current_action_space, gym.spaces.Discrete):
            #     # Reshape in case of discrete action
            #     actions = actions.reshape(-1, 1)
            # print("Rollout actions shape", rollout.actions.shape)

            # next_obss_t = obs_as_tensor(next_obss, self.ppo.device)
            with torch.no_grad():
                # Compute value for the last timestep
                next_obss_values = self.ppo.policy.predict_values(
                    rollout.next_observations)

            # TODO: not sure if it's needed, but we can vectorize it 
            # for idx, done in enumerate(rollout.dones):
                # FIXME: how do we get access to truncated data from TimeLimit
                # truncated = False
                # if done and truncated:
                    # rollout.rewards[idx] += self.ppo.gamma * \
                        # next_obss_values[idx].item()

            # add to ppo's buffer
            # TODO: can we swap sb3 buffer for ours in rollout? 
            # here we're limited to adding a single transition at a time, while we can have multiple timesteps here already
            self.ppo.rollout_buffer.add(
                rollout.observations,
                rollout.actions,
                rollout.rewards,
                self.last_dones,
                self.last_values,
                self.last_log_probs,
            )
            self.last_dones = rollout.dones

            if self.steps_since_last_train >= self.ppo.n_steps:
                # print(rollout.dones.cpu().numpy().shape, rollout.dones.cpu().numpy().dtype)
                # TODO: minor inconvenience, sb3 wants a numpy array to perform bool operation 1-dones (not supported in torch)
                self.ppo.rollout_buffer.compute_returns_and_advantage(
                    last_values=next_obss_values, dones=rollout.dones.cpu().numpy()
                )

                self.iteration += 1

                self.ppo.logger.record("time/iterations", self.iteration)
                self.ppo.logger.record(
                    "time/total_timesteps", self.num_timesteps)
                self.ppo.logger.dump(step=self.num_timesteps)

                self._before_training_iteration()
                self.ppo.train()
                self._after_training_iteration()

                self.ppo.policy.set_training_mode(False)
                self.steps_since_last_train = 0
                self.ppo.rollout_buffer.reset()

    def evaluate_exp(self):
        # TODO: this can be refactored in some nicer way
        def get_action(obs, task_label=None):
            return self.sample_rollout_action(obs)
        self.model.get_action = get_action
        return super().evaluate_exp()



def main():
    from avalanche_rl.benchmarks.generators.rl_benchmark_generators import (
        gym_benchmark_generator,
    )
    import torch
    from gym.spaces import Box

    num_envs = 1

    scenario = gym_benchmark_generator(
        ["CartPole-v1"],
        n_parallel_envs=num_envs,
        eval_envs=["CartPole-v1"],
        n_experiences=1,
    )

    # TODO: this must be wrapped nicely
    model_class: BasePolicy = get_policy_from_name(
        ActorCriticPolicy, "MlpPolicy")

    obs_space, action_space = scenario.envs[0].observation_space, Discrete(2)
    model = model_class( 
            obs_space,
            action_space,
            # TODO: this is not great
            lambda _: 1e-3,
            # use_sde=self.use_sde,
            # **self.policy_kwargs  # pytype:disable=not-instantiable
    )

    device = torch.device("cpu")

    # NOTE: model must not depend on experience and the action space for the agent must be decided before-hand

    # dummy_env = gym.make("CartPole-v1")

    strategy = SB3PPOStrategy(
        model,
        per_experience_steps=Timestep(10000),
        # training_experience_limit=Timestep(10_000, TimestepUnit.STEPS),
        # eval_experience_limit=Timestep(10, TimestepUnit.EPISODES),
        # periodic_eval_every=Timestep(1000, TimestepUnit.STEPS),
        device=device,
        eval_episodes=10
    )

    print("Starting experiment...")
    # results = [strategy.eval(scenario.test_stream)]
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Env ", experience.env)
        print("Current Task", experience.task_label, type(experience.task_label))
        strategy.train(experience, scenario.test_stream)
        # results.append(strategy.eval(scenario.test_stream))

    print("Training completed")
    print(strategy.eval(scenario.test_stream))


if __name__ == "__main__":
    main()
