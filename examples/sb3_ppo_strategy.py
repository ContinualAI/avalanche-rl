from avalanche_rl.benchmarks.rl_benchmark import RLExperience
from avalanche_rl.training.strategies import RLBaseStrategy
from avalanche_rl.training.strategies.rl_base_strategy import Timestep
from avalanche_rl.training.plugins.strategy_plugin import RLPlugin
from avalanche_rl.training.strategies.buffers import Rollout
from typing import Any, List, Dict, Sequence, Union
from torch.optim import Optimizer
import torch.nn as nn
import stable_baselines3 as sb3
from stable_baselines3 import PPO
import numpy as np
import torch


class SB3PPOStrategy(RLBaseStrategy):
    def __init__(
        self,
        model: nn.Module,
        per_experience_steps: Union[int, Timestep, List[Timestep]],
        sb3_policy: str = "MlpPolicy",
        rollouts_per_step: int = 1,
        max_steps_per_rollout: int = -1,
        updates_per_step: int = 1,
        device="cpu",
        max_grad_norm=None,
        plugins: Sequence[RLPlugin] = ...,
        discount_factor: float = 0.99,
        evaluator=...,
        eval_every=-1,
        eval_episodes: int = 1,
    ):
        super().__init__(
            model,
            None,
            per_experience_steps,
            None,
            rollouts_per_step,
            max_steps_per_rollout,
            updates_per_step,
            device,
            max_grad_norm,
            plugins,
            discount_factor,
            evaluator,
            eval_every,
            eval_episodes,
        )
        self.ppo: PPO = None
        self.sb3_policy = sb3_policy

    # TODO: optimizer step and loss computation all happen within ppo.train(), train_exp must be modified
    # TODO: num_envs to comply with sb3 checks on vec env
    # TODO: n_epochs must always be set to 1 to control it from outside
    # TODO: we can still use vectorized env since we're filling experiences mem from outside
    # TODO: logger integration
    # TODO: decide whether we keep current `choose_action` interface
    # TODO: transpose image as env wrappers
    def _before_training_exp(self, **kwargs):
        self.ppo = PPO(
            policy=policy,
            env=self.environment.unwrapped,
            # seed=rng_seed % (2**32),
            # verbose=0,
            device=device,
            n_epochs=1,
            **algorithm_kwargs,
        )
        self.ppo._setup_learn(np.inf, eval_env=None)
        return super()._before_training_exp(**kwargs)

    def train_exp(self, experience: RLExperience, eval_streams, **kwargs):
        self.environment = experience.environment
        self.n_envs = experience.n_envs
        # TODO:  keep track in default evaluator
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
        # self.model_adaptation()
        self.make_optimizer()

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

                # Backward
                self.optimizer.zero_grad()
                self._before_backward(**kwargs)
                self.loss.backward()
                self._after_backward(**kwargs)

                # Gradient norm clipping
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm)

                # Optimization step
                self._before_update(**kwargs)
                self.optimizer.step()
                self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
            # periodic evaluation
            self._periodic_eval(eval_streams, do_final=False)

        self.total_steps += self.rollout_steps
        self.environment.close()

        # Final evaluation
        self._periodic_eval(eval_streams, do_final=(
            self.timestep % self.eval_every != 0))
        self._after_training_exp(**kwargs)
