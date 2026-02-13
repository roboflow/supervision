# Contributing to Supervision 🛠️

Thank you for your interest in contributing to Supervision!

We are actively improving this library to reduce the amount of work you need to do to solve common computer vision problems.

## Code of Conduct

Please read and adhere to our [Code of Conduct](https://supervision.roboflow.com/latest/code_of_conduct/). This document outlines the expected behavior for all participants in our project.

## Table of Contents

- [Contribution Guidelines](#contribution-guidelines)
    - [Contributing Features](#contributing-features)
- [How to Contribute Changes](#how-to-contribute-changes)
- [Installation for Contributors](#installation-for-contributors)
- [Code Style and Quality](#code-style-and-quality)
    - [Pre-commit tool](#pre-commit-tool)
    - [Docstrings](#docstrings)
    - [Type checking](#type-checking)
- [Documentation](#documentation)
- [Cookbooks](#cookbooks)
- [Tests](#tests)
- [License](#license)

## Contribution Guidelines

We welcome contributions to:

1. Add a new feature to the library (guidance below).
2. Improve our documentation and add examples to make it clear how to leverage the supervision library.
3. Report bugs and issues in the project.
4. Submit a request for a new feature.
5. Improve our test coverage.

### Contributing Features ✨

Supervision is designed to provide generic utilities to solve problems. Thus, we focus on contributions that can have an impact on a wide range of projects.

For example, counting objects that cross a line anywhere on an image is a common problem in computer vision, but counting objects that cross a line 75% of the way through is less useful.

Before you contribute a new feature, consider submitting an Issue to discuss the feature so the community can weigh in and assist.

## How to Contribute Changes

First, fork this repository to your own GitHub account. Click "fork" in the top corner of the `supervision` repository to get started:

![Forking the repository](https://media.roboflow.com/fork.png)

![Creating a repository fork](https://media.roboflow.com/create_fork.png)

Then, run `git clone` to download the project code to your computer.

You should also set up `roboflow/supervision` as an "upstream" remote (that is, tell git that the reference Supervision repository was the source of your fork of it):

```bash
git remote add upstream https://github.com/roboflow/supervision.git
git fetch upstream
```

Move to a new branch using the `git checkout` command:

```bash
git checkout -b <scope>/<your_branch_name> upstream/develop
```

The name you choose for your branch should describe the change you want to make and start with an appropriate prefix:

- `feat/`: for new features (e.g., `feat/line-counter`)
- `fix/`: for bug fixes (e.g., `fix/memory-leak`)
- `docs/`: for documentation changes (e.g., `docs/update-readme`)
- `chore/`: for routine tasks, maintenance, or tooling changes (e.g., `chore/update-dependencies`)
- `test/`: for adding or modifying tests (e.g., `test/add-unit-tests`)
- `refactor/`: for code refactoring (e.g., `refactor/simplify-algorithm`)

Make any changes you want to the project code, then run the following commands to commit your changes:

```bash
git add -A
git commit -m "feat: add line counter functionality"
git push -u origin <your_branch_name>
```

Use conventional commit messages to clearly describe your changes. The format is:

```
<type>[optional scope]: <description>
```

Common types include:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools and libraries

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
5. Please share a Google Colab with minimal code to test a new feature or reproduce the issue whenever possible. Please ensure that Google Colab can be accessed without any restrictions.

When you submit your Pull Request, you will be asked to sign a Contributor License Agreement (CLA) by the `cla-assistant` GitHub bot. We can only respond to PRs from contributors who have signed the project CLA.

All pull requests will be reviewed by the maintainers of the project. We will provide feedback and ask for changes if necessary.

PRs must pass all tests and linting requirements before they can be merged.

## Installation for Contributors

Before starting your work on the project, set up your development environment:

1. **Clone your fork of the project:**

    **Option A: Recommended for most contributors (shallow clone of develop branch):**

    ```bash
    git clone --depth 1 -b develop https://github.com/YOUR_USERNAME/supervision.git
    cd supervision
    ```

    Replace `YOUR_USERNAME` with your GitHub username.

    > **Note**: Using `--depth 1` creates a shallow clone with minimal history and `-b develop` ensures you start with the development branch. This significantly reduces download size while providing everything needed to contribute.

    **Option B: Full repository clone (if you need complete history):**

    ```bash
    git clone https://github.com/YOUR_USERNAME/supervision.git
    cd supervision
    git checkout develop
    ```

2. **Set up the upstream remote:**

    ```bash
    git remote add upstream https://github.com/roboflow/supervision.git
    git fetch upstream
    ```

3. **Create and activate a virtual environment:**

    **On Linux/macOS:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

    **On Windows:**

    ```cmd
    python -m venv .venv
    .venv\Scripts\activate
    ```

4. **Install `uv`:**

    Follow the instructions on the [uv installation page](https://docs.astral.sh/uv/getting-started/installation/).

5. **Install project dependencies:**

    ```bash
    uv pip install -r pyproject.toml --group dev --group docs --extra metrics
    ```

6. **Verify the setup:**

    ```bash
    uv run pytest
    ```

## 🎨 Code Style and Quality

### Pre-commit tool

This project uses the [pre-commit](https://pre-commit.com/) tool to maintain code quality and consistency. Before submitting a pull request or making any commits, it is important to run the pre-commit tool to ensure that your changes meet the project's guidelines.

Furthermore, we have integrated a pre-commit GitHub Action into our workflow. This means that with every pull request opened, the pre-commit checks will be automatically enforced, streamlining the code review process and ensuring that all contributions adhere to our quality standards.

To run the pre-commit tool, follow these steps:

1. **Install pre-commit** (already included if you followed the installation steps above):

    ```bash
    uv sync --group dev
    ```

2. **Navigate to the project's root directory** (if not already there).

3. **Run pre-commit checks**:

    ```bash
    uv run pre-commit run --all-files
    ```

    This will execute the pre-commit hooks configured for this project. If any issues are found, the pre-commit tool will provide feedback on how to resolve them. Make the necessary changes and re-run the command until all issues are resolved.

4. **Install pre-commit as a git hook** (optional but recommended):

    ```bash
    uv run pre-commit install
    ```

    This will automatically run pre-commit checks every time you make a `git commit`.

### Docstrings

All new functions and classes in `supervision` should include docstrings. This is a prerequisite for any new functions and classes to be added to the library.

`supervision` adheres to the [Google Python docstring style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods). Please refer to the style guide while writing docstrings for your contribution.

### Type checking

Currently, there is no systematic type checking with mypy implemented in the project. This is a known limitation that may be addressed in future updates.

## 📝 Documentation

The `supervision` documentation is stored in a folder called `docs`. The project documentation is built using `mkdocs`.

To run the documentation locally:

1. **Install documentation dependencies** (if not already installed):

    ```bash
    uv sync --group docs
    ```

2. **Start the documentation server**:

    ```bash
    uv run mkdocs serve
    ```

3. **Access the documentation** at `http://127.0.0.1:8000` in your browser.

You can learn more about mkdocs on the [mkdocs website](https://www.mkdocs.org/).

## 🧑‍🍳 Cookbooks

We are always looking for new examples and cookbooks to add to the `supervision`
documentation. If you have a use case that you think would be helpful to others, please
submit a PR with your example. Here are some guidelines for submitting a new example:

- Create a new notebook in the [`docs/notebooks`](https://github.com/roboflow/supervision/tree/develop/docs/notebooks) folder.
- Add a link to the new notebook in [`docs/theme/cookbooks.html`](https://github.com/roboflow/supervision/blob/develop/docs/theme/cookbooks.html). Make sure to add the path to the new notebook, as well as a title, labels, author and supervision version.
- Use the [Count Objects Crossing the Line](https://supervision.roboflow.com/develop/notebooks/count-objects-crossing-the-line/) example as a template for your new example.
- Pin the version of `supervision` you are using in the notebook.
- Place an appropriate "Open in Colab" button at the top of the notebook. You can find an example of such a button in the aforementioned `Count Objects Crossing the Line` cookbook.
- **Notebook should be self-contained**. If you rely on external data (videos, images, etc.) or libraries, include download and installation commands in the notebook.
- Annotate the code with appropriate comments, including links to the documentation describing each of the tools you have used.

## 🧪 Tests

[`pytest`](https://docs.pytest.org/en/7.1.x/) is used to run our tests.

To run tests:

```bash
uv run pytest
```

To run tests with coverage:

```bash
uv run pytest --cov=supervision
```

## 🔍 PR Review Guidelines

These guidelines help reviewers provide consistent, actionable feedback and help maintainers make informed merge decisions efficiently.

### Review Objectives

Your primary goal as a reviewer is to:

1. **Validate PR completeness** against project requirements
2. **Identify risks** that could impact users or maintainability
3. **Provide actionable feedback** the author can immediately act upon
4. **Highlight quality gaps** in code, tests, or documentation

### 1. Overall Recommendation

Start your review with a clear, actionable recommendation:

- 🟢 **Approve** — Ready to merge as-is
- 🟡 **Minor Suggestions** — Minor improvements recommended but not blocking
- 🟠 **Request Changes** — Significant issues must be addressed before merge
- 🔴 **Block** — Critical issues require major rework

**Example:**

```
🟠 Request Changes — Missing unit tests for new `PolygonMerger` class and no documentation entry added for autogeneration.
```

### 2. PR Completeness Checklist

Verify the PR meets project requirements. Mark each item:

- ✅ **Complete** — Properly addressed
- ⚠️ **Incomplete** — Partially done, needs improvement
- ❌ **Missing** — Not provided
- 🔵 **N/A** — Not applicable to this PR

#### Required Items

- [ ] **Clear description** — What changed and why
- [ ] **Type of change** — Bug fix, feature, docs, etc.
- [ ] **Motivation/context** — Problem being solved (links to issue if relevant)
- [ ] **Changes list** — Summary of modifications
- [ ] **Tests** — Unit tests added/updated
- [ ] **Documentation** — Docstrings follow [Google-style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
- [ ] **Docs entry** — Added to mkdocs for autogeneration (new functions/classes only)
- [ ] **Google Colab** — Provided for demonstrating feature/fix (if applicable)
- [ ] **Screenshots/videos** — Included for visual changes (if applicable)

**Call out missing items explicitly:**

```
❌ Missing:
- Documentation entry not added to mkdocs navigation
- No unit tests provided for `merge_polygons()` function
```

### 3. Quality Assessment

#### 3.1 Code Quality

Provide **specific feedback using inline comments** on the changed code. Use **n/5** scoring:

- **5/5** 🟢 Excellent — Well-structured, idiomatic, no issues
- **4/5** 🟢 Good — Minor improvements possible
- **3/5** 🟡 Acceptable — Some issues to address
- **2/5** 🟠 Needs Work — Multiple problems
- **1/5** 🔴 Poor — Significant refactoring required

**Score: n/5** — [Brief justification]

**Check for:**

1. **Correctness**

    - Logic errors or edge cases not handled
    - Potential bugs (None checks, array bounds, division by zero)
    - Incorrect assumptions

2. **Python Best Practices**

    - Non-idiomatic patterns
    - Improper exception handling
    - Inefficient implementations
    - Missing or incorrect type hints

3. **Project Conventions**

    - **Docstrings:** Must follow [Google-style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
    - **Code style:** Must pass linting (`uv run pre-commit run --all-files`)
    - **Imports:** Standard library → third-party → local
    - **Naming:** Clear, descriptive, follows PEP 8

**Place inline comments directly on problematic code**, then reference them in your summary.

#### 3.2 Testing Quality

Use **n/5** scoring for test coverage and quality:

- **5/5** 🟢 Comprehensive — All cases covered, high-quality assertions
- **4/5** 🟢 Good — Most cases covered
- **3/5** 🟡 Adequate — Basic coverage, some gaps
- **2/5** 🟠 Insufficient — Major gaps
- **1/5** 🔴 Missing — No tests or tests don't validate functionality

**Score: n/5** — [Brief justification]

**For New Features or Bug Fixes:**

1. **Coverage Requirements**

    - [ ] Unit tests added for new functions/classes
    - [ ] Edge cases covered (empty inputs, None, large arrays, boundary conditions)
    - [ ] Regression tests for bug fixes

2. **Test Quality**

    - [ ] Assertions are specific (not just "no exception raised")
    - [ ] Tests use realistic scenarios
    - [ ] Test names clearly describe what they validate

#### 3.3 Documentation Quality

Use **n/5** scoring for documentation completeness:

- **5/5** 🟢 Excellent — Complete, clear, with good examples
- **4/5** 🟢 Good — Minor improvements possible
- **3/5** 🟡 Adequate — Basic docs present
- **2/5** 🟠 Insufficient — Incomplete or unclear
- **1/5** 🔴 Missing — No documentation

**Score: n/5** — [Brief justification]

**For New Features:**

1. **Docstring Requirements**

    - [ ] Docstrings for all public functions/classes
    - [ ] Parameters, return values, and exceptions documented
    - [ ] Usage examples in docstrings

2. **Documentation Integration**

    - [ ] Entry added to appropriate docs page (e.g., `docs/detection/tools/*.md`)
    - [ ] Added to mkdocs navigation (`mkdocs.yml`)
    - [ ] Changelog entry (`docs/changelog.md`) for user-facing changes

**For Changes to Existing Features:**

1. **Update Requirements**
    - [ ] Docstrings updated to reflect changes
    - [ ] Deprecated features marked with warnings
    - [ ] Migration guide for breaking changes

### 4. Risk Assessment

**Explicitly flag any risks with severity:**

- **5/5** 🔴 Critical — Blocks release, must fix
- **4/5** 🟠 High — Serious concern, should fix
- **3/5** 🟡 Medium — Notable risk, consider fixing
- **2/5** 🟢 Low — Minor concern
- **1/5** 🟢 Negligible — No real risk

**Risk Categories:**

1. **Breaking Changes**

    - Changes to public APIs (function signatures, return types)
    - Removal of deprecated features
    - Changed behavior in existing functionality
    - **If breaking:** Must include migration instructions

2. **Performance Impact**

    - Inefficient algorithms (O(n²) where O(n) possible)
    - Memory-intensive operations on large arrays
    - Potential bottlenecks in hot paths

3. **Compatibility Issues**

    - New Python version requirements
    - New dependencies
    - Platform-specific code

4. **Security Concerns**

    - Unvalidated user input
    - Potential code execution risks
    - Sensitive data exposure

**Example:**

```
Risk Level: 4/5 🟠 High Performance Risk

Nested loop detected - see inline comment in `zone.py` for vectorization suggestion.
```

### 5. Providing Constructive Suggestions

**Add inline comments to the code using GitHub's review interface**, then provide **suggested changes** using GitHub suggestion format:

````markdown
```suggestion
if detections is None or detections.mask is None:
    return None
return process(detections.mask)
```
````

**Suggestion Categories:**

1. **Code Improvements**

    - Logic simplifications
    - Better error handling
    - More readable implementations

2. **Performance Optimizations**

    - NumPy vectorization opportunities
    - Caching expensive computations
    - Batch processing

3. **Architecture Improvements**

    - Code reuse opportunities
    - Better abstractions
    - More maintainable designs

### 6. Review Summary Template

Use this structure for your final review comment:

```markdown
## Review Summary

### Recommendation
[emoji] [Status] — [One-sentence justification]

### PR Completeness
- ✅ Complete: [list key items]
- ❌ Missing: [list critical gaps]

### Quality Scores
- **Code Quality:** n/5 [emoji] — [brief reason]
- **Testing:** n/5 [emoji] — [brief reason]
- **Documentation:** n/5 [emoji] — [brief reason]

### Risk Level: n/5 [emoji]
[Brief risk description with reference to inline comments if applicable]

### Critical Issues (Must Fix)
1. [Issue description] — See comment on `file.py`
2. [Another blocking issue] — See comment on `test_file.py`

### Suggestions (Nice to Have)
1. [Improvement idea] — See suggestion on `file.py`
2. [Another optional enhancement]

### Next Steps for Author
1. [Clear action item]
2. [Another clear action item]
```

### Best Practices for Effective Reviews

**DO:**

1. ✅ **Use n/5 scoring** for quick assessment of quality dimensions
2. ✅ **Place comments directly on code** using GitHub's inline comment feature
3. ✅ **Use GitHub suggestion format** for code changes when possible
4. ✅ **Reference inline comments** in your summary (e.g., "See comment on `file.py:function()`")
5. ✅ **Explain *why*** something is a problem, not just *what* is wrong
6. ✅ **Distinguish** between blocking issues and nice-to-haves
7. ✅ **Acknowledge** good work and clever solutions
8. ✅ **Run linter** locally if needed: `uv run pre-commit run --all-files`

**DON'T:**

1. ❌ **Don't mention line numbers** in summary — place comments inline instead
2. ❌ **Don't give vague feedback** like "improve code quality"
3. ❌ **Don't nitpick** on personal style preferences (defer to automated tools)
4. ❌ **Don't assume** the author knows project conventions
5. ❌ **Don't focus only on problems** — recognize what's good
6. ❌ **Don't let perfect** be the enemy of good (minor issues shouldn't block useful PRs)

### Review Workflow

1. **Review files** in the PR, placing inline comments on specific issues
2. **Use GitHub suggestions** for concrete code improvements
3. **Draft your summary** using the template above
4. **Reference inline comments** instead of mentioning specific line numbers
5. **Submit review** with clear recommendation and next steps

### Tone and Communication

- **Be respectful and constructive** — Contributors are volunteers
- **Be specific and technical** — Help them learn
- **Be pragmatic** — Balance ideal vs. practical
- **Be consistent** — Follow these guidelines every time

**Remember:** Your goal is to help maintainers efficiently assess PRs and help contributors improve their work. Focus on **actionable feedback** that moves the PR toward merge.

## 📄 License

By contributing, you agree that your contributions will be licensed under an [MIT license](../LICENSE.md).
