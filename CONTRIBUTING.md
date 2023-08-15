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

First, fork this repository to your own GitHub account. Click "fork" in the top corner of the `supervision` repository to get started:

![Forking the repository](https://media.roboflow.com/fork.png)

![Creating a repository fork](https://media.roboflow.com/create_fork.png)

Then, run `git clone` to download the project code to your computer.

Move to a new branch using the `git checkout` command:

```bash
git checkout -b <your_branch_name>
```

The name you choose for your branch should describe the change you want to make (i.e. `line-counter-docs`).

Make any changes you want to the project code, then run the following commands to commit your changes:

```bash
git add .
git commit -m "Your commit message"
git push -u origin main
```

This project utilizes the [pre-commit](https://pre-commit.com/) tool to maintain code quality and consistency. Before submitting a pull request or making any commits, it is important to run the pre-commit tool to ensure that your changes meet the project's guidelines.

To run the pre-commit tool, follow these steps:

1. Install pre-commit by running the following command: `poetry install`. It will not only install pre-commit but also install all the deps and dev-deps of project

2. Once pre-commit is installed, navigate to the project's root directory.

3. Run the command `pre-commit run --all-files`. This will execute the pre-commit hooks configured for this project against the modified files. If any issues are found, the pre-commit tool will provide feedback on how to resolve them. Make the necessary changes and re-run the pre-commit command until all issues are resolved.

4. You can also install pre-commit as a git hook by execute `pre-commit install`. Every time you made `git commit` pre-commit run automatically for you.

Then, go back to your fork of the `supervision` repository, click "Pull Requests", and click "New Pull Request".

![Opening a pull request](https://media.roboflow.com/open_pr.png)

Make sure the `base` branch is `develop` before submitting your PR.

On the next page, review your changes then click "Create pull request":

![Configuring a pull request](https://media.roboflow.com/create_pr_submit.png)

Next, write a description for your pull request, and click "Create pull request" again to submit it for review:

![Submitting a pull request](https://media.roboflow.com/write_pr.png)

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
