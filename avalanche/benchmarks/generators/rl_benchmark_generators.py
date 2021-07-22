from avalanche.benchmarks.rl_benchmark import *
import gym
from gym import envs
from typing import *
import numpy as np
import importlib.util


def get_all_envs_id():
    all_envs = envs.registry.all()
    return [env_spec.id for env_spec in all_envs]


def make_env(env_name: str, env_kwargs: Dict[Any, Any] = dict()):
    return gym.make(env_name, **env_kwargs)


def gym_benchmark_generator(
        env_names: List[str] = [],
        environments: List[gym.Env] = [],
        n_random_envs: int = None, 
        env_kwargs: Dict[str, Dict[Any, Any]] = {},
        n_parallel_envs: int = 1,
        eval_envs: Union[List[str], List[gym.Env]] = None,
        n_experiences=None,
        envs_ids_to_sample_from: List[str] = None, *args, **kwargs) -> RLScenario:
    # three ways to create environments from gym
    if env_names is not None and len(env_names):
        envs_ = [gym.make(ename, **env_kwargs.get(ename, {}))
                 for ename in env_names]
    elif environments is not None and len(environments):
        envs_ = environments
    elif n_random_envs is not None and n_random_envs > 0:
        # choose `n_envs` random envs either from the registered ones or from the provided ones
        to_choose = get_all_envs_id(
        ) if envs_ids_to_sample_from is not None else envs_ids_to_sample_from
        ids = np.random.choice(to_choose, n_random_envs)  
        envs_ = [gym.make(ename, **env_kwargs.get(ename, {})) for ename in ids]
    else:
        raise ValueError(
            'You must provide at least one argument among `env_names`,`environments`, `n_random_envs`!')

    # eval environments: if not provided, the same training environments are also used for testing
    if eval_envs is None:
        eval_envs = envs_
    elif len(eval_envs) and type(eval_envs[0]) is str:
        eval_envs = [gym.make(ename, **env_kwargs.get(ename, {}))
                     for ename in eval_envs]
        # TODO: delayed feature
        # if a list of env names is provided, we don't build the enviornment until actual evaluation occurs
        # eval_envs = [
        #     lambda: gym.make(ename, **env_kwargs.get(ename, {}))
        #     for ename in eval_envs]
    elif len(eval_envs) and not isinstance(eval_envs[0], gym.Env):
        raise ValueError(
            "Unrecognized type for `eval_envs`, make sure to pass a list of environments or of environment names.")

    # envs will be cycled through if n_experiences > len(envs_) otherwise only the first n will be used
    n_experiences = n_experiences if n_experiences is not None else len(envs_)         
    # # per_experience_episodes variable number of episodes depending on env
    # if type(per_experience_episodes) is list:
    #     assert len(per_experience_episodes) == n_experiences

    return RLScenario(
        envs=envs_, n_experiences=n_experiences,
        n_parallel_envs=n_parallel_envs, eval_envs=eval_envs, *args, **kwargs)


def atari_benchmark_generator() -> RLScenario:
    # TODO: setup env wrappers
    pass


# check AvalancheLab extra dependency
if importlib.util.find_spec('continual_habitat_lab') is not None and importlib.util.find_spec('habitat_sim') is not None:
    from continual_habitat_lab import ContinualHabitatLabConfig, ContinualHabitatEnv
    from continual_habitat_lab.scene_manager import SceneManager
    from avalanche.training.strategies.reinforcement_learning.rl_base_strategy import TimestepUnit, Timestep

    def _compute_experiences_length(
            task_change_steps: int, scene_change_steps: int, n_exps: int,
            unit: TimestepUnit) -> List[Timestep]:
        # change experience every time either task or scene changes (might happen at non-uniform timesteps) 
        change1 = min(task_change_steps, scene_change_steps)
        change2 = max(task_change_steps, scene_change_steps) - change1

        # those two timestep will alternate
        steps = [Timestep(change1, unit), Timestep(change2, unit)]
        return [steps[i % 2] for i in range(n_exps)]

    def habitat_benchmark_generator(
            cl_habitat_lab_config: ContinualHabitatLabConfig,
            # eval_config: ContinualHabitatLabConfig = None, 
            max_steps_per_experience: int = 1000,
            change_experience_on_scene_change: bool = False,
            *args, **kwargs) -> Tuple[RLScenario, List[Timestep]]:

        # number of experiences as the number of tasks defined in configuration
        n_exps = len(cl_habitat_lab_config.tasks)

        # compute number of steps per experience
        steps_per_experience = Timestep(max_steps_per_experience)
        task_len_in_episodes = cl_habitat_lab_config.task_iterator.get('max_task_repeat_episodes', -1)
        task_len_in_steps = cl_habitat_lab_config.task_iterator.get('max_task_repeat_steps', -1)

        if task_len_in_episodes > 0:
            steps_per_experience = Timestep(
                task_len_in_episodes, TimestepUnit.EPISODES)
        elif task_len_in_steps > 0:
            steps_per_experience = Timestep(
                task_len_in_steps, TimestepUnit.STEPS)

        # whether to account every change of scene as a new experience (scene_path inhibits scene change)
        if change_experience_on_scene_change and cl_habitat_lab_config.scene.scene_path is None:
            # count number of scenes
            sm = SceneManager(cl_habitat_lab_config)
            for _, scenes in sm._scenes_by_dataset.items():
                n_exps += len(scenes)
            scene_change_in_episodes = cl_habitat_lab_config.scene.max_scene_repeat_episodes
            scene_change_in_steps = cl_habitat_lab_config.scene.max_scene_repeat_steps
            if scene_change_in_episodes > 0:
                assert steps_per_experience.unit == TimestepUnit.EPISODES, "Step unit between simulator and avalanche setting must match!"
                steps_per_experience = _compute_experiences_length(
                    steps_per_experience.value, scene_change_in_episodes,
                    n_exps, steps_per_experience.unit)
            if scene_change_in_steps > 0:
                assert steps_per_experience.unit == TimestepUnit.STEPS, "Step unit between simulator and avalanche setting must match!"
                steps_per_experience = _compute_experiences_length(
                    steps_per_experience.value, scene_change_in_steps, n_exps,
                    steps_per_experience.unit)
        else:
            steps_per_experience = [steps_per_experience] * n_exps

        # instatiate RLScenario from a given lab configuration
        env = ContinualHabitatEnv(cl_habitat_lab_config)
        # TODO: evaluating on same env changes its state, must have some evaluation mode
        # if eval_config is None:
            # eval_env = env

        # also return computed number of steps per experience
        # NOTE: parallel_env only supported on distributed settings due to opengl lock
        # TODO: test with multiple gpus?
        return RLScenario(
            envs=[env],
            n_experiences=n_exps, n_parallel_envs=1, eval_envs=[env],
            *args, **kwargs), steps_per_experience
