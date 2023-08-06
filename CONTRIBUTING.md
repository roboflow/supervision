# Contributing to Supervision 🛠️

Thank you for your interest in contributing to Supervision!

We are actively improving this library to reduce the amount of work you need to do to solve common computer vision problems.

## Contribution Guidelines

We welcome contributions to:

1. Add a new feature to the library (guidance below).
2. Improve our documentation and add examples to make it clear how to leverage the supervision library.
3. Report bugs and issues in the project.
4. Submit a request for a new feature.
5. Improve our test coverage.

### Contributing Features

Supervision is designed to provide generic utilities to solve problems. Thus, we focus on contributions that can have an impact on a wide range of projects.

For example, counting objects that cross a line anywhere on an image is a common problem in computer vision, but counting objects that cross a line 75% of the way through is less useful.

Before you contribute a new feature, consider submitting an Issue to discuss the feature so the community can weigh in and assist.

## How to Contribute Changes

First, fork this repository to your own GitHub account. Create a new branch that describes your changes (i.e. `line-counter-docs`). Push your changes to the branch on your fork and then submit a pull request to `develop` branch of this repository.

When creating new functions, please ensure you have the following:

1. Docstrings for the function and all parameters.
2. Unit tests for the function.
3. Examples in the documentation for the function.
4. Created an entry in our docs to autogenerate the documentation for the function.
5. Please share google colab with minimal code to test new feature or reproduce PR whenever it is possible. Please ensure that google colab can be accessed without any issue. 

All pull requests will be reviewed by the maintainers of the project. We will provide feedback and ask for changes if necessary.

PRs must pass all tests and linting requirements before they can be merged.

## 🧹 code quality 

We provide two handy commands inside the `Makefile`, namely:

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)

So far, **there is no types checking with mypy**. See [issue](https://github.com/roboflow-ai/template-python/issues/4). 

## 🧪 tests 

[`pytests`](https://docs.pytest.org/en/7.1.x/) is used to run our tests.
