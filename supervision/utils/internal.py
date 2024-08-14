import functools
import inspect
import os
import warnings
from typing import Any, Callable, Generic, Optional, Set, TypeVar


class SupervisionWarnings(Warning):
    """Supervision warning category.
    Set the deprecation warnings visibility for Supervision library.
    You can set the environment variable SUPERVISON_DEPRECATION_WARNING to '0' to
    disable the deprecation warnings.
    """

    pass


def format_warning(msg, category, filename, lineno, line=None):
    """
    Format a warning the same way as the default formatter, but also include the
    category name in the output.
    """
    return f"{category.__name__}: {msg}\n"


warnings.formatwarning = format_warning

if os.getenv("SUPERVISON_DEPRECATION_WARNING") == "0":
    warnings.simplefilter("ignore", SupervisionWarnings)
else:
    warnings.simplefilter("always", SupervisionWarnings)


def warn_deprecated(message: str):
    """
    Issue a warning that a function is deprecated.

    Args:
        message (str): The message to display when the function is called.
    """
    warnings.warn(message, category=SupervisionWarnings, stacklevel=2)


def deprecated_parameter(
    old_parameter: str,
    new_parameter: str,
    map_function: Callable = lambda x: x,
    warning_message: str = "Warning: '{old_parameter}' in '{function_name}' is "
    "deprecated: use '{new_parameter}' instead.",
    **message_kwargs,
):
    """
    A decorator to mark a function's parameter as deprecated and issue a warning when
    used.

    Parameters:
        old_parameter (str): The name of the deprecated parameter.
        new_parameter (str): The name of the parameter that should be used instead.
        map_function (Callable): A function used to map the value of the old
            parameter to the new parameter. Defaults to the identity function.
        warning_message (str): The warning message to be displayed when the
            deprecated parameter is used. Defaults to a generic warning message with
            placeholders for the old parameter, new parameter, and function name.
        **message_kwargs: Additional keyword arguments that can be used to customize
            the warning message.

    Returns:
        Callable: A decorator function that can be applied to mark a function's
            parameter as deprecated.

    Examples:
        ```python
        @deprecated_parameter(
            old_parameter=<OLD_PARAMETER_NAME>,
            new_parameter=<NEW_PARAMETER_NAME>
        )
        def example_function(<NEW_PARAMETER_NAME>):
            pass

        # call function using deprecated parameter
        example_function(<OLD_PARAMETER_NAME>=<OLD_PARAMETER_VALUE>)
        ```
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if old_parameter in kwargs:
                if args and hasattr(args[0], "__class__"):
                    class_name = args[0].__class__.__name__
                    function_name = f"{class_name}.{func.__name__}"
                else:
                    function_name = func.__name__

                warn_deprecated(
                    message=warning_message.format(
                        function_name=function_name,
                        old_parameter=old_parameter,
                        new_parameter=new_parameter,
                        **message_kwargs,
                    )
                )

                kwargs[new_parameter] = map_function(kwargs.pop(old_parameter))

            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated(reason: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_deprecated(f"{func.__name__} is deprecated: {reason}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


T = TypeVar("T")


class classproperty(Generic[T]):
    """
    A decorator that combines @classmethod and @property.
    It allows a method to be accessed as a property of the class,
    rather than an instance, similar to a classmethod.

    Usage:
        @classproperty
        def my_method(cls):
            ...
    """

    def __init__(self, fget: Callable[..., T]):
        """
        Args:
            The function that is called when the property is accessed.
        """
        self.fget = fget

    def __get__(self, owner_self: Any, owner_cls: Optional[type] = None) -> T:
        """
        Override the __get__ method to return the result of the function call.

        Args:
            owner_self: The instance through which the attribute was accessed, or None.
                Irrelevant for class properties.
            owner_cls: The class through which the attribute was accessed.

        Returns:
            The result of calling the function stored in 'fget' with 'owner_cls'.
        """
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(owner_cls)


def get_instance_variables(instance: Any, include_properties=False) -> Set[str]:
    """
    Get the public variables of a class instance.

    Args:
        instance (Any): The instance of a class
        include_properties (bool): Whether to include properties in the result

    Usage:
        ```python
        detections = Detections(xyxy=np.array([1,2,3,4]))
        variables = get_class_variables(detections)
        # ["xyxy", "mask", "confidence", ..., "data"]
        ```
    """
    if isinstance(instance, type):
        raise ValueError("Only class instances are supported, not classes.")

    fields = set(
        (
            name
            for name, val in inspect.getmembers(instance)
            if not callable(val) and not name.startswith("_")
        )
    )

    if not include_properties:
        properties = set(
            (
                name
                for name, val in inspect.getmembers(instance.__class__)
                if isinstance(val, property)
            )
        )
        fields -= properties

    return fields
