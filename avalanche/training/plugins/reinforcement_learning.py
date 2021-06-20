from collections import defaultdict

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.dqn.policies import DQNPolicy
from avalanche.benchmarks.rl_benchmark import RLExperience
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy, get_policy_from_name
from typing import Callable, List, Union, Any, Dict, Sequence, Optional
from gym import Env
import torch.nn as nn
import torch.optim as optim
from avalanche.training import default_logger

# essentially we're augmenting strategies and plugins with additional callbacks
# based on RL training structure


# class RLPlugin(StrategyPlugin):

#     def __init__(self, per_experience_episodes: Union[int, List[int]] = 1,):
#         super().__init__()
#         self.per_experience_episodes = per_experience_episodes

#     def get_training_action(self):
#         raise NotImplementedError()
#     # TODO: vectorized env


# class DQNPlugin(RLPlugin):

#     def __init__(
#             self, per_experience_episodes: int = 1, stable_policy: str = None, *
#             args, **kwargs):
#         super().__init__(per_experience_episodes)
#         self.model = None

#     def before_training_exp(self, strategy: BaseStrategy, **kwargs):
#         env = strategy.experience.environment
#         if self.model is None:
#             self.model = DQN(policy='MlpPolicy', env=env, **kwargs)
#         else:
#             self.model.env = env
#         return super().before_training_exp(strategy, **kwargs)


class AvlSB3Callback(BaseCallback):
    # TODO: pre-defined logging callbacks from AVL
    def __init__(
            self, callback_stack: Dict[str, List[Callable[[Any],
                                                          Any]]],
            verbose: int = 0):
        super().__init__(verbose=verbose)
        self.callback_stack = callback_stack

    def _call_callback_stack(self, callback_name: str):
        # call functions and return values
        return [callback() for callback in self.callback_stack[callback_name]]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self._call_callback_stack('on_training_start')

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self._call_callback_stack('on_rollout_start')

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        values = self._call_callback_stack('on_step')
        # return false if any value in values is false
        if not all(values):
            return False
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self._call_callback_stack('on_rollout_end')

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self._call_callback_stack('on_training_end')

    @staticmethod
    def from_avalanche_strategy(
            base_rl_strategy: 'RLStrategy', cl_strategy: BaseStrategy = None,
            plugins: Sequence['StrategyPlugin'] = [], verbose: int = 0):
        """
        Maps avalanche strategy callbacks to those of sb3.
        Args:
            cl_strategy (BaseStrategy): [description]
        """
        callback_map = {
            'on_training_start': ['before_training'],
            'on_step': ['before_update', 'after_training_epoch'],
            'on_training_end': ['after_training'],
            }
        cl_foo = []
        callback_stack = defaultdict(list)
        for sb3_call, avl_calls in callback_map.items():
            rl_foo = lambda: getattr(base_rl_strategy, sb3_call)()
            if cl_strategy is not None:
                cl_foo = [
                    lambda: getattr(cl_strategy, acall)(cl_strategy)
                    for acall in avl_calls]
            plugin_foo = [
                lambda: getattr(p, acall)(base_rl_strategy)
                for acall in avl_calls for p in plugins]

            callback_stack[sb3_call].extend([rl_foo] + cl_foo + plugin_foo)
        return AvlSB3Callback(callback_stack, verbose=verbose)


# FIXME: how can you make it so people can modify stuff easily?
# TODO: probably can't access loss, re-implement algorithms..? 
# TODO: self.loss with custom update_locals..?
class RLStrategy(BaseStrategy):

    def __init__(
            self, model: Union[str, nn.Module], 
            envs: List[Env],
            policy_type: Union[str, BasePolicy],
            cl_strategy: BaseStrategy,
            per_experience_episodes: int,
            train_mb_size: int = 1,
            eval_mb_size: int = 1,
            device='cpu',
            plugins: Optional[Sequence['StrategyPlugin']] = [],
            optimizer_class: optim.Optimizer = optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            custom_baseline_alg: BaseAlgorithm = None,  # TODO: allow custom alg
            evaluator=default_logger, 
            eval_every=-1,
            ):

        if type(policy_type) is str:
            policy_type = self._map_policy_type(policy_type)
        if type(model) is str:
            self.policy_class = get_policy_from_name(policy_type, model)

        self.policy_kwargs = dict(
            optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs)

        self.policy = DQN(
            policy=self.policy_class, env=envs[0],
            policy_kwargs=self.policy_kwargs, verbose=1)
        print("Model", self.policy_model)
        # optimizer can be passed to sb3 
        super().__init__(
            model=self.policy_model, optimizer=self.policy_optimizer,
            criterion=None, train_mb_size=train_mb_size,
            train_epochs=per_experience_episodes, eval_mb_size=eval_mb_size,
            device=device, plugins=plugins, evaluator=evaluator,
            eval_every=eval_every)

        # map strategy callbacks to sb3 hooks
        self.per_experience_episodes = self.train_epochs
        self.sb3_callback = AvlSB3Callback.from_avalanche_strategy(
            self, cl_strategy, plugins)

    def _map_policy_type(self, pname: str):
        # TODO:
        if pname == 'dqn':
            return DQNPolicy

    # loss/criterion is wired-in in the alg code 
    @property
    def policy_optimizer(self):
        # return optimizer from sb3 algorithm implementation (or custom algorithm if provided)
        return self.policy.policy.optimizer

    @property
    def policy_model(self):
        return self.policy.policy.q_net

    # TODO: AVL dataset online generation to support strategies like Replay
    # TODO: remove unneeded methods

    # rlstrategy callbacks are not mapped, use default sb3 callbacks here

    def on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def learn(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    # FIXME: merge stable baseline 3 callbacks into avl 
    # https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

    def train(
            self, experiences: Union[RLExperience, Sequence[RLExperience]],
            eval_streams:
            Optional[Sequence[Union[RLExperience, Sequence[RLExperience]]]] = None,
            **kwargs):
        return super().train(experiences, eval_streams=eval_streams, **kwargs)

    def train_exp(self, experience: RLExperience, eval_streams, **kwargs):
        self.experience = experience
        self.model.train()

        # Data Adaptation (e.g. add new samples/data augmentation)
        # self.before_train_dataset_adaptation(**kwargs)
        # self.train_dataset_adaptation(**kwargs)
        # self.after_train_dataset_adaptation(**kwargs)
        # self.make_train_dataloader(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        self.model_adaptation()
        self.make_optimizer()

        self.before_training_exp(**kwargs)

        env = experience.environment
        # FIXME: dqn, ppo, a2c.. using policy type
        # self.policy = DQN(policy='MlpPolicy', env=env,
        #                   policy_kwargs=self.policy_kwargs)
        # episode instead of epoch
        # self.episode = 0
        # FIXME: use n_episodes or switch to steps? 
        self.policy.learn(
            total_timesteps=self.per_experience_episodes * 10,
            callback=self.sb3_callback)
        # self.before_training_epoch(**kwargs)

        # self.training_epoch(**kwargs)
        # self.after_training_epoch(**kwargs)
        # self._periodic_eval(eval_streams, do_final=False)

        # Final evaluation
        # self._periodic_eval(eval_streams, do_final=do_final)
        # self.after_training_exp(**kwargs)
