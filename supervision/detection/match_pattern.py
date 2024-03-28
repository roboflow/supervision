import itertools
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np

from supervision.detection.core import Detections
from supervision.geometry.core import Position


class _Constraint:
    """
    A constraint is a rule that a pattern must follow. It is defined
    by a function and its arguments. The arguments are strings that specify an object
    of the pattern and one of its fields.
    For example, this constraint tests that the objects A and B of your pattern have
    the same class:
    ```python
    _Constraint(lambda x, y: x == y, "A.class_id", "B.class_id")
    ```

    !!! tip

        You can use a value instead of a function as the criteria. It will check that
        the arguments are all equal to this value. For instance, this constraint tests
        that the object A of your pattern has a class_id equal to 1.
        ```python
        _Constraint(1, "A.class_id")
        ```
        This works with any number of arguments, so you can check several objects at
        once:
        ```python
        _Constraint(1, "A.class_id", "B.class_id")
        ```
    """

    def __init__(
        self, criteria: Union[Callable[..., bool], Any], arguments: List[str]
    ) -> None:
        """
        Args:
            criteria (Callable): A function that takes N arguments and returns a
                boolean. Criteria can also be any value, in which case the constraint
                checks that every argument is equal to this value.
            *arguments (str): A list of N strings that will be given as arguments for
                the criteria. The arguments should look like "name.field". The name of
                the object can be any name that doesn't contain a dot (`.`). The field
                should be one of the following:
                 - `xyxy`, `mask`, `class_id`, `confidence`, or `tracker_id`
                 - one of the `Position` enum strings
                 - a field from the `data` attribute of your detections
        """
        validate_arguments(arguments)
        self.arguments = arguments
        if callable(criteria):
            self.criteria = criteria
        else:
            self.criteria = lambda *args: all(equality(arg, criteria) for arg in args)


def validate_arguments(arguments: List[str]) -> None:
    for argument in arguments:
        if argument.count(".") != 1:
            raise ValueError(
                f"Constraint argument should be `name.field`, got: '{argument}'"
            )


def equality(arg1, arg2):
    if isinstance(arg1, np.ndarray) or isinstance(arg2, np.ndarray):
        return (arg1 == arg2).all()
    return arg1 == arg2


class MatchPattern:
    """
    A pattern is a set of constraints that apply to detections. You can think of
    patterns as regex for detections. `MatchPattern` will return all matches that
    satisfy all the constraints.

    A pattern is described as named boxes organized according to rules. Each rule is
    given as a constraint. For instance "BoxA and BoxB should have the same class",
    "Boxes A and B should overlap", etc. The constraints are functions that apply to
    fields from the detections (the `class_id`, the `xyxy` coordinates, etc.).

    For example, if you want to search for a cat and a dog that have the same center
    point you can use the following pattern:
    ```python
    import cv2
    import supervision as sv
    from ultralytics import YOLO

    image = cv2.imread(<SOURCE_IMAGE_PATH>)
    model = YOLO('yolov8s.pt')

    pattern = sv.MatchPattern(
        [
            (lambda class_id: class_id == 0, ["Cat.class_id"]),  # class_id for cat is 0
            (1, ["Dog.class_id"]),  # class_id for dog is 1
            (
                lambda dog_center, cat_center: dog_center == cat_center),
                ["Dog.CENTER", "Cat.CENTER"]
            ),
        ]
    )

    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    matches = pattern.match(detections)
    ```

    This will return all the matches that satisfy the constraints. The result is a list
    of `Detections`. A field `match_name` is added to the Detections.data to keep
    track of the names in your pattern.
    ```python
    first_match = matches[0]
    first_match["match_name"]  # ["Cat", "Dog"]
    ```
    """

    def __init__(
        self,
        constraints: List[Tuple[Union[Callable[..., bool], Any], List[str]]],
    ):
        """
        Args:
            constraints (List[Tuple[Union[Callable[..., bool], Any], List[str]]]):
                A list of constraints. Each constraint contains a criterion and a list
                of arguments:
                 - `criteria` is a function that returns a boolean value. See
                 `_Constraint` for more information.
                 - arguments is a list of strings. Each argument is composed of
                `name.field`. The field should be one of the following:
                 - `xyxy`, `mask`, `class_id`, `confidence`, or `tracker_id`
                 - one of the `Position` enum strings
                 - a field from the `data` attribute of your detections
        """
        self._constraints: List[_Constraint] = []
        for constraint in constraints:
            criteria, arguments = constraint
            self.add_constraint(criteria, arguments)

    def add_constraint(
        self, criteria: Union[Callable[..., bool], Any], arguments: List[str]
    ) -> None:
        """
        Adds a constraint to the matching pattern.
        Args:
            criteria: A function that returns a boolean value or any value you want to
                match with the `arguments`. See `_Constraint` for more details.
            arguments: A list of strings. See `_Constraint` for more details.
        """
        self._constraints.append(_Constraint(criteria, arguments))

    def match(self, detections: Detections) -> List[Detections]:
        """
        Matches the pattern of the constraints to the detections.

        Args:
            detections (Detections): Detections to match the pattern with.

        Returns:
            List[Detections]: List of detections that match the constraints. A specific
            field `match_name` is added to the matches to keep track of the names
            specified in the pattern arguments.
        """
        combinations = self._generate_combinations(len(detections))

        names = self._get_names_from_constraints()
        index = 0
        while index < len(combinations):
            combination = dict(zip(names, combinations[index]))
            template_kwargs = {
                name: detections[int(box_index)]
                for name, box_index in combination.items()
            }
            for constraint in self._constraints:
                criteria_args = [
                    _get_argument(template_kwargs, detections, arg)
                    for arg in constraint.arguments
                ]
                if not constraint.criteria(*criteria_args):
                    incompatible_boxes = {
                        arg_name: combination[arg_name]
                        for arg_name in self._get_names_from_arguments(
                            constraint.arguments
                        )
                    }
                    filter_bool = np.ones(len(combinations), dtype=bool)
                    for name, values in incompatible_boxes.items():
                        filter_bool &= combinations[name] == values
                    combinations = combinations[~filter_bool]
                    break
            else:
                index += 1

        results: List[Detections] = []
        for valid_combination in combinations:
            indexes = list(valid_combination)
            matching_boxes: Detections = detections[indexes]  # type: ignore
            matching_boxes["match_name"] = names
            results.append(matching_boxes)

        return results

    def _get_names_from_constraints(self) -> List[str]:
        """
        Returns the object names used in the constraints.
        """
        arguments = [
            arg for constraint in self._constraints for arg in constraint.arguments
        ]
        return self._get_names_from_arguments(arguments)

    def _get_names_from_arguments(self, arguments: Iterable[str]) -> List[str]:
        """
        Returns the object names used in the arguments. Sorted and unique.
        """
        return sorted(
            list({arg.split(".")[0] if "." in arg else arg for arg in arguments})
        )

    def _generate_combinations(self, num_detections) -> np.ndarray:
        """
        Generates all the possible combinations for the pattern matching.
        Returns an array of shape (N, M) where N is the number of combinations and M is
        the number of objects in the pattern. Each row corresponds to the set of indexes
        from detections.
        """
        names = self._get_names_from_constraints()
        return np.fromiter(
            itertools.permutations(range(num_detections), len(names)),
            np.dtype([(name, int) for name in names]),
        )


def _get_argument(kwargs: Dict[str, Any], detections: Detections, argument: str) -> Any:
    name, subfield = argument.split(".")
    if subfield in ["xyxy", "mask", "class_id", "confidence", "tracker_id"]:
        return getattr(kwargs[name], subfield)[0]
    if subfield in Position.list():
        return kwargs[name].get_anchors_coordinates(Position[subfield])[0]
    if subfield in detections.data:
        return kwargs[name][subfield][0]
    raise ValueError(f"Unknown field '{subfield}' for object '{name}'")
