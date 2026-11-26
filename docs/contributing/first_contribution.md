# Your First Contribution

This page walks through a complete contribution, from fork to pull request. For the full environment, style and testing rules, see the [Development guide](development.md).

!!! warning "Before you open a pull request"

    Sign the Contributor License Agreement (CLA) when the `cla-assistant` bot asks you to. It is required: we can only respond to pull requests from contributors who have signed.

## 1. Fork the repository

Fork `roboflow/supervision` into your **personal** GitHub account (click "Fork" in the top corner of the repository). Avoid organization-owned forks: maintainers cannot push to their pull request branches, which makes small fixes and rebases slower for everyone.

![Forking the repository](https://media.roboflow.com/fork.png)

## 2. Clone and set up the upstream remote

A shallow clone of `develop` is enough for most contributions:

```bash
git clone --depth 1 -b develop https://github.com/YOUR_USERNAME/supervision.git
cd supervision
git remote add -t develop upstream https://github.com/roboflow/supervision.git
git fetch upstream
```

Replace `YOUR_USERNAME` with your GitHub username. Drop `--depth 1` if you need the full history.

## 3. Install the project

```bash
uv sync --group dev --group docs --extra metrics
uv run pytest
```

If the tests pass, your environment works. More detail is in the [Development guide](development.md#set-up-your-environment).

## 4. Create a branch

Branch from `upstream/develop`, with a prefix that describes the change:

```bash
git checkout -b <prefix>/<short-description> upstream/develop
```

- `feat/`: new features (for example `feat/line-counter`)
- `fix/`: bug fixes (for example `fix/memory-leak`)
- `docs/`: documentation changes (for example `docs/update-readme`)
- `chore/`: maintenance or tooling changes
- `test/`: adding or modifying tests
- `refactor/`: code refactoring without behavior changes

## 5. Make the change and check it

When you add a new function, include:

1. Docstrings for the function and all parameters.
2. Unit tests for the function.
3. A usage example in the documentation.
4. An entry in the docs so the API reference is generated.
5. A Google Colab with minimal code that tests the feature or reproduces the bug, whenever possible. Make sure it opens without any access restrictions.

Before you commit, run the checks locally:

```bash
uv run pre-commit run --all-files
uv run pytest
```

## 6. Commit and push

Stage the files you changed and use a conventional commit message, `<type>[optional scope]: <description>`:

```bash
git add <changed files>
git commit -m "feat: add line counter functionality"
git push -u origin <prefix>/<short-description>
```

Common types are `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test` and `chore`.

## 7. Open the pull request

Go to your fork on GitHub, click "Pull requests", then "New pull request". Set the `base` branch to `develop`, fill in the description template, and create the pull request.

![Opening a pull request](https://media.roboflow.com/open_pr.png)

Then:

- Sign the CLA when the bot asks.
- Keep **Allow edits by maintainers** enabled.
- Wait for CI. All tests and linting checks must pass before a pull request can be merged.

A maintainer reviews every pull request, gives feedback and may ask for changes.
