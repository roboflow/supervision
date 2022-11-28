# Python Template 🐍
A template repo holding our common setup for a python project.

## Installation

You can install the package using pip

```bash
pip install -e .
```

or for development

```bash
pip install -e ".[dev]"
```

## Structure

The project has the following structure

```
├── .github
│   └── workflows
│       └── test.yml # holds our github action config 
├── .gitignore
├── Makefile
├── README.md
├── setup.py
├── src
│   ├── __init__.py 
│   ├── hello.py 
└── test 
    └── test_hello.py
```

### Code Quality 🧹

We provide two handy commands inside the `Makefile`, namely:

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)

So far, **there is no types checking with mypy**. See [issue](https://github.com/roboflow-ai/template-python/issues/4). 

### Tests 🧪

[`pytests`](https://docs.pytest.org/en/7.1.x/) is used to run our tests.

### Publish on PyPi 🚀

**Important**: Before publishing, edit `__version__` in [src/__init__](/src/__init__.py) to match the wanted new version.

We use [`twine`](https://twine.readthedocs.io/en/stable/) to make our life easier. You can publish by using

```
export PYPI_USERNAME="you_username"
export PYPI_PASSWORD="your_password"
export PYPI_TEST_PASSWORD="your_password_for_test_pypi"
make publish -e PYPI_USERNAME=$PYPI_USERNAME -e PYPI_PASSWORD=$PYPI_PASSWORD -e PYPI_TEST_PASSWORD=$PYPI_TEST_PASSWORD
```

You can also use token for auth, see [pypi doc](https://pypi.org/help/#apitoken). In that case,

```
export PYPI_USERNAME="__token__"
export PYPI_PASSWORD="your_token"
export PYPI_TEST_PASSWORD="your_token_for_test_pypi"
make publish -e PYPI_USERNAME=$PYPI_USERNAME -e PYPI_PASSWORD=$PYPI_PASSWORD -e PYPI_TEST_PASSWORD=$PYPI_TEST_PASSWORD
```

**Note**: We will try to push to [test pypi](https://test.pypi.org/) before pushing to pypi, to assert everything will work

### CI/CD 🤖

We use [GitHub actions](https://github.com/features/actions) to automatically run tests and check code quality when a new PR is done on `main`.

On any pull request, we will check the code quality and tests.

When a new release is created, we will try to push the new code to PyPi. We use [`twine`](https://twine.readthedocs.io/en/stable/) to make our life easier. 

The **correct steps** to create a new realease are the following:
- edit `__version__` in [src/__init__](/src/__init__.py) to match the wanted new version.
- create a new [`tag`](https://git-scm.com/docs/git-tag) with the release name, e.g. `git tag v0.0.1 && git push origin v0.0.1` or from the GitHub UI.
- create a new release from GitHub UI

The CI will run when you create the new release.

# Q&A

## Why no cookiecutter?
This is a template repo, it's meant to be used inside GitHub upon repo creation.

## Why reinvent the wheel?

There are several very good templates on GitHub, I prefer to use code we wrote instead of blinding taking the most starred template and having features we don't need. From experience, it's better to keep it simple and general enough for our specific use cases.