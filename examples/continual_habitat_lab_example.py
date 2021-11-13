from continual_habitat_lab.utils import suppress_habitat_logging
suppress_habitat_logging()
from habitat_sim.logging import logger
logger.propagate = False
from avalanche.training.strategies.reinforcement_learning.rl_base_strategy import RLBaseStrategy
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from gym.core import ObservationWrapper
from avalanche.models.dqn import ConvDeepQN
from avalanche.models.actor_critic import ActorCriticMLP, ConvActorCritic
from avalanche.training.strategies.reinforcement_learning.actor_critic import A2CStrategy

from avalanche.benchmarks.generators.rl_benchmark_generators import habitat_benchmark_generator
from continual_habitat_lab import ContinualHabitatLabConfig, ContinualHabitatEnv
from omegaconf import OmegaConf
from torch.optim import Adam
import gym
from gym.spaces.box import Box
from gym.wrappers.time_limit import TimeLimit
# TODO: insert into utils


class HabitatObservations(ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old_space = env.observation_space['rgba']
        self.observation_space = Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=[3, *old_space.shape[:2]],
            dtype=old_space.dtype,
        ) 

    def observation(self, observation):
        # select rgb and make it channel first
        # TODO: why is collided in here..?
        # print('obs', observation.keys())
        return observation['rgba'][..., :3].transpose(2, 0, 1)


class HabitatPlugin(StrategyPlugin):
    def __init__(self, max_steps: int = None):
        super().__init__()
        self.max_steps = max_steps

    def before_make_env(self, strategy: RLBaseStrategy, **kwargs):
        strategy.environment = HabitatObservations(strategy.environment)
        # put some time limit here
        strategy.environment = TimeLimit(
            strategy.environment, max_episode_steps=self.max_steps)
        super().before_make_env(strategy)


image_resolution = (128, 128)
# pass max steps to get sensible episode termination (also needed to display stats)
max_steps_per_ep = 100
if __name__ == "__main__":

    config = {'tasks': [{'type': 'ObjectNav', 'name': 'Task0'}], 
              'scene': {
            'scene_path': '/home/nick/datasets/habitat/replicav1/room_2/habitat/mesh_semantic.ply'
        },
        'agent': {
            'sensor_specifications': [{
                'type': "RGBA",
                'resolution': image_resolution
            }]
        }, 
        # 'task_iterator': {
        # 'max_steps'
        # }
    }

    cfg = ContinualHabitatLabConfig(OmegaConf.create(config), from_cli=False)

    scenario, steps_per_exps = habitat_benchmark_generator(
        cfg, max_steps_per_experience=100000)

    # default actions: turn right-turn left-move forward
    # FIXME: try with normal network not from atari
    model = ConvActorCritic(3, image_resolution, 3)
    print("Model", model)

    optimizer = Adam(model.parameters(), lr=1e-4)
    print("Steps per experience", steps_per_exps)
    strategy = A2CStrategy(
        model, optimizer, per_experience_steps=steps_per_exps,
        max_steps_per_rollout=5, device='cuda:0', eval_every=-1,
        plugins=[HabitatPlugin(max_steps_per_ep)])

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Env ", experience.env)
        strategy.train(experience, [])
        print('Training completed')
        # TODO:
        # print('Computing accuracy on the whole test set')
        # results.append(strategy.eval(scenario.test_stream))

    # rmean, rstd, lengths = evaluate(model, n_episodes=100, device=device)
    # print(f"Reward mean/std: {rmean}, {rstd}. Episode lengths: {lengths}")

    scenario.envs[0].close()
