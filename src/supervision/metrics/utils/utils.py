def ensure_pandas_installed() -> None:
    try:
        import pandas  # noqa
    except ImportError:
        raise ImportError(
            "`metrics` extra is required to run the function."
            " Run `uv pip install 'supervision[metrics]'` or"
            " `uv add supervision --extra metrics`."
        )
