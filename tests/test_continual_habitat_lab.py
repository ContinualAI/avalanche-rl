from avalanche_rl.training.strategies.rl_base_strategy import TimestepUnit
import pytest
try:
    from continual_habitat_lab import ContinualHabitatLabConfig, ContinualHabitatEnv
except ImportError:
    pytest.skip("Need Continual Habitat Lab to run these tests!", allow_module_level=True)
from avalanche_rl.benchmarks.generators.rl_benchmark_generators import habitat_benchmark_generator
from omegaconf import OmegaConf

def make_cfg(max_steps = 100, res=(128, 128), task_iterator_max_eps=None):
    config = {'tasks': [{'type': 'VoidTask', 'max_steps': max_steps, 'name': f'Task{i}'} for i in range(2)], 
        'scene': {
            'scene_path': '/home/nick/datasets/habitat/replicav1/room_2/habitat/mesh_semantic.ply'
        }, 
        'agent': {
            'sensor_specifications': [{
                'type': "RGBA",
                'resolution': res
            }]
        }
    }
    if task_iterator_max_eps is not None:
        config.update({'task_iterator': {'max_task_repeat_episodes': 2}})

    return ContinualHabitatLabConfig(OmegaConf.create(config), from_cli=False)


def test_simple_creation():
    cfg = make_cfg(100)
    scenario, steps_per_exps = habitat_benchmark_generator(cfg, max_steps_per_experience=100)
    assert len(steps_per_exps) == 2 and steps_per_exps[0] == steps_per_exps[1] and steps_per_exps[0].value==100
    # assert number of tasks equal n_envs
    assert len(scenario.envs) == 2
    assert isinstance(scenario.envs[0], ContinualHabitatEnv)
    scenario.envs[0].close()

def test_reset():
    # observation resolution
    resolution = (64, 64)

    cfg = make_cfg(100, resolution)
    scenario, steps_per_exps = habitat_benchmark_generator(cfg, max_steps_per_experience=100)
    env = scenario.envs[0] 
    obs = env.reset()
    assert 'rgba' in obs 
    # RGBA shape
    assert obs['rgba'].shape == (*resolution, 4)

    scenario.envs[0].close()

def test_task_iterator():
    cfg = make_cfg(max_steps=10, task_iterator_max_eps=2)
    scenario, steps_per_exps = habitat_benchmark_generator(cfg, max_steps_per_experience=10)
    assert steps_per_exps[0]==steps_per_exps[1] and steps_per_exps[0].value == 2 and steps_per_exps[0].unit == TimestepUnit.EPISODES

    # check whether task actually changes on new experience
    tasks = []
    for experience in scenario.train_stream:
        env:ContinualHabitatEnv = experience.environment
        assert isinstance(env, ContinualHabitatEnv)

        # play 2 episodes
        for ep_no in range(2):
            # NOTE: task changes on reset!
            env.reset()
            assert env.task_iterator._active_task_idx == experience.current_experience
            tasks.append(env.current_task.name)

            while not env.done:
                env.step(env.action_space.sample())

        print(env.tasks, cfg.task_iterator, env.task_iterator._active_task_idx)

    tasks = set(tasks)
    assert len(tasks) == 2 

    scenario.envs[0].close()
