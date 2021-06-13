from avalanche.benchmarks.scenarios.generic_definitions import Experience
from avalanche.benchmarks.scenarios.generic_cl_scenario import TGenericScenarioStream, GenericScenarioStream, TGenericCLScenario, Iterable, GenericCLScenario, GenericExperience
from gym.core import Env
from typing import Dict, Union, Optional, Sequence, Any, List


def rl_experience_factory(
        stream: GenericScenarioStream, exp_idx:
        Union[int, slice, Iterable[int]]) -> 'RLExperience':
    # simple things first
    return RLExperience(stream, exp_idx, stream.scenario.envs[exp_idx])


# NCScenario equivalent (which subclasses GenericCLScenario)
class RLScenario(GenericCLScenario['RLExperience']):

    def __init__(self, envs: List[Env],
                 n_experiences: int,
                 task_labels: bool=True,
                 per_experience_episodes: int = 1,  # uniform to per_experience_classes format
                 #  shuffle: bool = True, shuffle task?
                 seed: Optional[int] = None,
                 fixed_task_order: Optional[Sequence[int]] = None,
                 class_ids_from_zero_from_first_exp: bool = False,
                 class_ids_from_zero_in_each_exp: bool = False,
                 reproducibility_data: Optional[Dict[str, Any]] = None):
        self.envs = envs
        # ignore labels for now
        stream_definitions = {
            'train': (self.envs, None, None),
            'test': (self.envs, None, None)
        }
        super().__init__(stream_definitions=stream_definitions,
                         complete_test_set_only=False,
                         experience_factory=rl_experience_factory)

    # FIXME: check which inherited method makes sense and which one does not work
    # FIXME: No notion of classes in experience

    def get_classes_timeline(self, current_experience: int,
                             stream: str = 'train'):
        """
        Returns the classes timeline given the ID of a experience.

        Given a experience ID, this method returns the classes in that
        experience, previously seen classes, the cumulative class list and a
        list of classes that will be encountered in next experiences of the
        same stream.

        Beware that by default this will obtain the timeline of an experience
        of the **training** stream. Use the stream parameter to select another
        stream.

        :param current_experience: The reference experience ID.
        :param stream: The stream name.
        :return: A tuple composed of four lists: the first list contains the
            IDs of classes in this experience, the second contains IDs of
            classes seen in previous experiences, the third returns a cumulative
            list of classes (that is, the union of the first two list) while the
            last one returns a list of classes that will be encountered in next
            experiences.
        """
        # inhibit this behavior by passing empty lists
        return ([], [], [], [])


# FIXME: slicing with changing environment 
# class GenericRLStream(GenericScenarioStream):
    # def __init__(self: TGenericScenarioStream, name: str, scenario: TGenericCLScenario, *, slice_ids: List[int]):
        # super().__init__(name, scenario, slice_ids=slice_ids)

# class GenericRLStream(GenericScenarioStream):
#     def __init__(self: TGenericScenarioStream,
#                  name: str,
#                  scenario: RLScenario,
#                  *,
#                  slice_ids: List[int] = None):
#         self.slice_ids: Optional[List[int]] = slice_ids
#         """
#         Describes which experiences are contained in the current stream slice. 
#         Can be None, which means that this object is the original stream. """

#         self.name: str = name
#         self.scenario = scenario

#     def __len__(self) -> int:
#         """
#         Gets the number of experiences this stream it's made of.

#         :return: The number of experiences in this stream.
#         """
#         if self.slice_ids is None:
#             return len(self.scenario.stream_definitions[self.name].exps_data)
#         else:
#             return len(self.slice_ids)

#     def __getitem__(self, exp_idx: Union[int, slice, Iterable[int]]) -> \
#             Union[TExperience, TScenarioStream]:
#         """
#         Gets a experience given its experience index (or a stream slice given
#         the experience order).

#         :param exp_idx: An int describing the experience index or an
#             iterable/slice object describing a slice of this stream.

#         :return: The experience instance associated to the given experience
#             index or a sliced stream instance.
#         """
#         if isinstance(exp_idx, int):
#             if exp_idx < len(self):
#                 if self.slice_ids is None:
#                     return self.scenario.experience_factory(self, exp_idx)
#                 else:
#                     return self.scenario.experience_factory(
#                         self, self.slice_ids[exp_idx])
#             raise IndexError('Experience index out of bounds' +
#                              str(int(exp_idx)))
#         else:
#             return self._create_slice(exp_idx)

#     def _create_slice(self: TGenericScenarioStream,
#                       exps_slice: Union[int, slice, Iterable[int]]) \
#             -> TScenarioStream:
#         """
#         Creates a sliced version of this stream.

#         In its base version, a shallow copy of this stream is created and
#         then its ``slice_ids`` field is adapted.

#         :param exps_slice: The slice to use.
#         :return: A sliced version of this stream.
#         """
#         stream_copy = copy.copy(self)
#         slice_exps = _get_slice_ids(exps_slice, len(self))

#         if self.slice_ids is None:
#             stream_copy.slice_ids = slice_exps
#         else:
#             stream_copy.slice_ids = [self.slice_ids[x] for x in slice_exps]
#         return stream_copy


class RLExperience(GenericExperience[RLScenario,
                                     GenericScenarioStream['RLExperience',
                                                           RLScenario]]):
    """
    Defines a "New Classes" experience. It defines fields to obtain the current
    dataset and the associated task label. It also keeps a reference to the
    stream from which this experience was taken.
    """

    def __init__(self,
                 origin_stream: GenericScenarioStream[
                     'RLExperience', RLScenario],
                 current_experience: int, env: Env):
        """
        Creates a ``NCExperience`` instance given the stream from this
        experience was taken and and the current experience ID.

        :param origin_stream: The stream from which this experience was
            obtained.
        :param current_experience: The current experience ID, as an integer.
        """
        super(RLExperience, self).__init__(origin_stream, current_experience)
        self.env = env

    @property
    def environment(self) -> Env:
        return self.env


if __name__ == "__main__":
    # create scenario
    import gym
    env = gym.make('CartPole-v1')
    # FIXME: you can pass single env but have multiple tasks (n_experiences), 
    # should at least return same env. To handle by higher level AtariGenerator
    rl_scenario = RLScenario([env], n_experiences=2, task_labels=True)
    print(rl_scenario.n_experiences)
    print(rl_scenario.original_test_dataset)
    # FIXME: works but no typing support
    tstream = rl_scenario.train_stream
    print(tstream, len(tstream))
    # print(rl_scenario.classes_in_experience) no sense?
    print('slicing', tstream[:1])
    for exp in tstream:
        print(exp, exp.current_experience)
        env = exp.environment
        obs = env.reset()
        print(obs)
