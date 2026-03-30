from __future__ import annotations

import functools
import inspect
import os
import warnings
from collections.abc import Callable
from typing import Any, Generic, TypeVar


class SupervisionWarnings(Warning):
    """Supervision warning category.
    Set the deprecation warnings visibility for Supervision library.
    You can set the environment variable SUPERVISON_DEPRECATION_WARNING to '0' to
    disable the deprecation warnings.
    """

    pass


def format_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    line: str | None = None,
) -> str:
    """
    Format a warning the same way as the default formatter, but also include the
    category name in the output.
    """
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = format_warning

if os.getenv("SUPERVISON_DEPRECATION_WARNING") == "0":
    warnings.simplefilter("ignore", SupervisionWarnings)
else:
    warnings.simplefilter("always", SupervisionWarnings)


def warn_deprecated(message: str) -> None:
    """
    Issue a warning that a function is deprecated.

    Args:
        message: The message to display when the function is called.
    """
    warnings.warn(message, category=SupervisionWarnings, stacklevel=2)


def deprecated_parameter(
    old_parameter: str,
    new_parameter: str,
    map_function: Callable[[Any], Any] = lambda x: x,
    warning_message: str = "Warning: '{old_parameter}' in '{function_name}' is "
    "deprecated: use '{new_parameter}' instead.",
    **message_kwargs: Any,
) -> Callable[[Any], Any]:
    """
    A decorator to mark a function's parameter as deprecated and issue a warning when
    used.

    Args:
        old_parameter: The name of the deprecated parameter.
        new_parameter: The name of the parameter that should be used instead.
        map_function: A function used to map the value of the old
            parameter to the new parameter. Defaults to the identity function.
        warning_message: The warning message to be displayed when the
            deprecated parameter is used. Defaults to a generic warning message with
            placeholders for the old parameter, new parameter, and function name.
        **message_kwargs: Additional keyword arguments that can be used to customize
            the warning message.

    Returns:
        A decorator function that can be applied to mark a function's
            parameter as deprecated.

    Examples:
        ```pycon
        >>> from supervision.utils.internal import deprecated_parameter
        >>> import warnings
        >>> @deprecated_parameter(
        ...     old_parameter='old_name',
        ...     new_parameter='new_name'
        ... )
        ... def example_function(new_name=None):
        ...     return new_name
        >>> # Calling with new parameter works normally
        >>> example_function(new_name='value')
        'value'
        >>> # Calling with old parameter triggers warning but still works
        >>> with warnings.catch_warnings(record=True):
        ...     result = example_function(old_name='deprecated_value')
        ...     print(result)
        deprecated_value

        ```
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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


def deprecated(reason: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(cls_or_func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.isclass(cls_or_func):
            original_init: Callable[..., None] = cls_or_func.__init__

            @functools.wraps(original_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warn_deprecated(f"{cls_or_func.__name__} is deprecated: {reason}")
                original_init(self, *args, **kwargs)

            cls_or_func.__init__ = new_init
            return cls_or_func
        else:

            @functools.wraps(cls_or_func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                warn_deprecated(f"{cls_or_func.__name__} is deprecated: {reason}")
                return cls_or_func(*args, **kwargs)

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

    def __get__(self, owner_self: Any, owner_cls: type | None = None) -> T:
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


def get_instance_variables(instance: Any, include_properties: bool = False) -> set[str]:
    """
    Get the public variables of a class instance.

    Args:
        instance: The instance of a class
        include_properties: Whether to include properties in the result

    Usage:
        ```pycon
        >>> from supervision.utils.internal import get_instance_variables
        >>> import numpy as np
        >>> from supervision import Detections
        >>> detections = Detections(xyxy=np.array([[1, 2, 3, 4]]))
        >>> variables = get_instance_variables(detections)
        >>> 'xyxy' in variables
        True
        >>> 'data' in variables
        True

        ```
    """
    if isinstance(instance, type):
        raise ValueError("Only class instances are supported, not classes.")

    fields = {
        name
        for name, val in inspect.getmembers(instance)
        if not callable(val) and not name.startswith("_")
    }

    if not include_properties:
        properties = {
            name
            for name, val in inspect.getmembers(instance.__class__)
            if isinstance(val, property)
        }
        fields -= properties

    return fields
