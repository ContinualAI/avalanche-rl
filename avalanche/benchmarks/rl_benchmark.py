from avalanche.benchmarks.scenarios.generic_definitions import Experience
from avalanche.benchmarks.scenarios.generic_cl_scenario import TGenericScenarioStream, GenericScenarioStream, TGenericCLScenario
from typing import runtime_checkable
from gym.core import Env
from typing import Dict, Union, Optional, Sequence, Any, List

@runtime_checkable
class RLExperience(Experience):

    def __init__(self) -> None:
        super().__init__()

    @property
    def environment(self)->Env:
        ...

    @property
    def env(self)->Env:
        return self.environment


class RLScenario:

    def __init__(self, envs: List[Env],
                 n_experiences: int,
                 task_labels: bool,
                 per_experience_episodes: int, # uniform to per_experience_classes format
                #  shuffle: bool = True, shuffle task?
                 seed: Optional[int] = None,
                 fixed_task_order: Optional[Sequence[int]] = None,
                 class_ids_from_zero_from_first_exp: bool = False,
                 class_ids_from_zero_in_each_exp: bool = False,
                 reproducibility_data: Optional[Dict[str, Any]] = None):
        pass


class GenericRLStream(GenericScenarioStream):
    def __init__(self: TGenericScenarioStream,
                 name: str,
                 scenario: TGenericCLScenario,
                 *,
                 slice_ids: List[int] = None):
        self.slice_ids: Optional[List[int]] = slice_ids
        """
        Describes which experiences are contained in the current stream slice. 
        Can be None, which means that this object is the original stream. """

        self.name: str = name
        self.scenario = scenario

    def __len__(self) -> int:
        """
        Gets the number of experiences this stream it's made of.

        :return: The number of experiences in this stream.
        """
        if self.slice_ids is None:
            return len(self.scenario.stream_definitions[self.name].exps_data)
        else:
            return len(self.slice_ids)

    def __getitem__(self, exp_idx: Union[int, slice, Iterable[int]]) -> \
            Union[TExperience, TScenarioStream]:
        """
        Gets a experience given its experience index (or a stream slice given
        the experience order).

        :param exp_idx: An int describing the experience index or an
            iterable/slice object describing a slice of this stream.

        :return: The experience instance associated to the given experience
            index or a sliced stream instance.
        """
        if isinstance(exp_idx, int):
            if exp_idx < len(self):
                if self.slice_ids is None:
                    return self.scenario.experience_factory(self, exp_idx)
                else:
                    return self.scenario.experience_factory(
                        self, self.slice_ids[exp_idx])
            raise IndexError('Experience index out of bounds' +
                             str(int(exp_idx)))
        else:
            return self._create_slice(exp_idx)

    def _create_slice(self: TGenericScenarioStream,
                      exps_slice: Union[int, slice, Iterable[int]]) \
            -> TScenarioStream:
        """
        Creates a sliced version of this stream.

        In its base version, a shallow copy of this stream is created and
        then its ``slice_ids`` field is adapted.

        :param exps_slice: The slice to use.
        :return: A sliced version of this stream.
        """
        stream_copy = copy.copy(self)
        slice_exps = _get_slice_ids(exps_slice, len(self))

        if self.slice_ids is None:
            stream_copy.slice_ids = slice_exps
        else:
            stream_copy.slice_ids = [self.slice_ids[x] for x in slice_exps]
        return stream_copy

