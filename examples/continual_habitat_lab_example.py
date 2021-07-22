from avalanche.training.strategies.reinforcement_learning.rl_base_strategy import RLBaseStrategy
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from gym.core import ObservationWrapper
from avalanche.models.dqn import ConvDeepQN
from avalanche.models.actor_critic import ActorCriticMLP
from avalanche.training.strategies.reinforcement_learning.actor_critic import A2CStrategy

from avalanche.benchmarks.generators.rl_benchmark_generators import habitat_benchmark_generator
from continual_habitat_lab import ContinualHabitatLabConfig, ContinualHabitatEnv
from omegaconf import OmegaConf
from torch.optim import Adam
import gym

# TODO: insert into utils


class HabitatObservations(ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def observation(self, observation):
        # select rgb
        print('obs', observation.keys())
        return observation['rgb'][..., :3]


class HabitatPlugin(StrategyPlugin):
    def __init__(self):
        super().__init__()

    def before_make_env(self, strategy: RLBaseStrategy, **kwargs):
        strategy.environment = HabitatObservations(strategy.environment)
        super().before_make_env(strategy)

# FIXME: make this work
image_resolution = (128, 128)
max_steps_per_ep = 1000
if __name__ == "__main__":

    config = {'tasks': [{'type': 'ObjectNav', 'max_steps': max_steps_per_ep, 'name': 'Task0'}], 
              'scene': {
            'scene_path': '/home/nick/datasets/habitat/replicav1/room_2/habitat/mesh_semantic.ply'
        },
        'agent': {
            'sensor_specifications': [{
                'type': "RGB",
                'resolution': image_resolution
            }]
        }
    }

    cfg = ContinualHabitatLabConfig(OmegaConf.create(config), from_cli=False)

    scenario, steps_per_exps = habitat_benchmark_generator(cfg)

    # default actions: turn right-turn left-move forward
    model = ConvDeepQN(3, image_resolution, 3)
    print("Model", model)

    optimizer = Adam(model.parameters(), lr=1e-4)
    print("Steps per experience", steps_per_exps)
    strategy = A2CStrategy(
        model, optimizer, per_experience_steps=steps_per_exps,
        max_steps_per_rollout=5, device='cuda:0', eval_every=-1,
        plugins=[HabitatPlugin()])

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
