try:
    import pyav
except ImportError:
    raise ImportError(
        "The pyav backend is not installed, please install it using `pip install supervision[video]`"
    )


class Backend:
    def __init__(self, source: str | int):
        raise NotImplementedError("The pyav backend is not implemented yet")
