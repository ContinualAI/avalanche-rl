import pytest
from avalanche_rl.benchmarks.rl_benchmark_generators \
    import atari_benchmark_generator, gym_benchmark_generator


def test_env_creation():
    scenario = gym_benchmark_generator(
        ['CartPole-v1'],
        n_experiences=1, n_parallel_envs=1, eval_envs=['CartPole-v1'])
    assert len(scenario.train_stream) == 1
    assert len(scenario.test_stream) == 1
    assert scenario.test_stream[0].environment.spec.id == \
        scenario.train_stream[0].environment.spec.id == \
        'CartPole-v1'


@pytest.mark.parametrize('n_exps', [1, 4, 16])
def test_task_label(n_exps):
    envs = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']
    scenario = gym_benchmark_generator(
        envs, n_experiences=n_exps, eval_envs=envs[:2])
    for exp_no in range(n_exps):
        assert scenario.train_stream[exp_no].task_label == (
            exp_no % scenario._num_original_envs)
        assert scenario.train_stream[exp_no].environment.spec.id == \
            envs[exp_no % len(envs)] 

    assert len(scenario.test_stream) == 2
    for exp_no in range(min(n_exps, 2)):
        assert scenario.test_stream[exp_no].task_label == exp_no
        assert scenario.test_stream[exp_no].environment.spec.id == envs[exp_no] 


@pytest.mark.parametrize('n_exps', [1, 4, 16])
def test_atari_task_label(n_exps):
    envs = ['PongNoFrameskip-v4',
            'BreakoutNoFrameskip-v4', 'FreewayNoFrameskip-v0']
    scenario = atari_benchmark_generator(
        envs, frame_stacking=True, normalize_observations=True,
        terminal_on_life_loss=True, n_experiences=n_exps, eval_envs=envs[: 2])
    for exp_no in range(n_exps):
        assert scenario.train_stream[exp_no].task_label == (
            exp_no % scenario._num_original_envs)
        assert scenario.train_stream[exp_no].environment.spec.id == \
            envs[exp_no % len(envs)] 

    assert len(scenario.test_stream) == 2
    for exp_no in range(min(n_exps, 2)):
        assert scenario.test_stream[exp_no].task_label == exp_no
        assert scenario.test_stream[exp_no].environment.spec.id == envs[exp_no] 
