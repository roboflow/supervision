def ensure_pandas_installed():
    try:
        import pandas  # noqa
    except ImportError:
        raise ImportError(
            "`metrics` extra is required to run the function."
            " Run `pip install 'supervision[metrics]'` or"
            " `poetry add supervision -E metrics`"
        )
