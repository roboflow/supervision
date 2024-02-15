import functools
import warnings
from typing import Callable


def deprecated_parameter(old_param: str, new_param: str, map_func: Callable = lambda x: x):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if old_param in kwargs:
                # In case of a method, display also the class name.
                if args and hasattr(args[0], "__class__"):
                    class_name = args[0].__class__.__name__
                    function_name = f"{class_name}.{func.__name__}"
                else:
                    function_name = func.__name__

                # Display deprecation warning
                warnings.warn(
                    f"Warning: '{old_param}' in '{function_name}' is deprecated: use '{new_param}' instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                # Map old_param to new_param
                kwargs[new_param] = map_func(kwargs.pop(old_param))

            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated(reason: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


class classproperty(property):
    """
    A decorator that combines @classmethod and @property.
    It allows a method to be accessed as a property of the class,
    rather than an instance, similar to a classmethod.

    Usage:
        @classproperty
        def my_method(cls):
            ...
    """

    def __get__(self, owner_self: object, owner_cls: type) -> object:
        """
        Override the __get__ method to return the result of the function call.

        Args:
        owner_self: The instance through which the attribute was accessed, or None.
        owner_cls: The class through which the attribute was accessed.

        Returns:
        The result of calling the function stored in 'fget' with 'owner_cls'.
        """
        return self.fget(owner_cls)
