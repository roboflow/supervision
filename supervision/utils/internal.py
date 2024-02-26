import functools
import warnings
from typing import Callable


def deprecated_parameter(
    old_parameter: str, 
    new_parameter: str, 
    map_function: Callable = lambda x: x, 
    warn_message: str = "Warning: '{old_parameter}' in '{function_name}' is deprecated: " \
                        "use '{new_parameter}' instead.",
    **message_kwargs
):
    """
    A decorator to mark a function's parameter as deprecated and issue a warning when used.

    Parameters:
    - old_parameter (str): The name of the deprecated parameter.
    - new_parameter (str): The name of the parameter that should be used instead.
    - map_function (Callable, optional): A function used to map the value of the old parameter to the new parameter.
      Defaults to the identity function.
    - warn_message (str, optional): The warning message to be displayed when the deprecated parameter is used.
      Defaults to a generic warning message with placeholders for the old parameter, new parameter, and function name.
    - **message_kwargs: Additional keyword arguments that can be used to customize the warning message.

    Returns:
    Callable: A decorator function that can be applied to mark a function's parameter as deprecated.

    Usage Example:
    ```python
    @deprecated_parameter(old_parameter="old_param", new_parameter="new_param")
    def example_function(new_param):
        print(f"Function called with new_param: {new_param}")

    # When calling the function with the deprecated parameter:
    example_function(old_param="deprecated_value")
    ```
    This will trigger a deprecation warning and execute the function with the mapped value of the deprecated parameter.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if old_parameter in kwargs:
                # In case of a method, display also the class name.
                if args and hasattr(args[0], "__class__"):
                    class_name = args[0].__class__.__name__
                    function_name = f"{class_name}.{func.__name__}"
                else:
                    function_name = func.__name__

                # Display deprecation warning
                warnings.warn(
                    message=warn_message.format(function_name=function_name, 
                                                old_parameter=old_parameter, 
                                                new_parameter=new_parameter, 
                                                **message_kwargs),
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                # Map old_param to new_param
                kwargs[new_parameter] = map_function(kwargs.pop(old_parameter))

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
