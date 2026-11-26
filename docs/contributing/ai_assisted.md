# AI-Assisted Contributions

AI coding assistants are welcome in this project, as long as the contribution that reaches a pull request is something you understand, have tested and are ready to maintain. This page describes how to set up an assistant and how to work with it. It applies to any tool, for example Claude Code, GitHub Copilot or Codex.

!!! warning "You stay the contributor"

    You are responsible for every line you submit, whoever or whatever wrote it. You are also the one who signs the [CLA](index.md), and you answer the review comments. Keep **Allow edits by maintainers** enabled on a pull request from your personal fork, as described in [Your First Contribution](first_contribution.md).

## Give the assistant the project guidance

The repository already ships instructions for AI tools. Make sure your assistant reads them before it changes code:

- [`AGENTS.md`](https://github.com/roboflow/supervision/blob/develop/AGENTS.md): architecture, design patterns, test structure, deprecation rules and the commands to run before a commit. [`CLAUDE.md`](https://github.com/roboflow/supervision/blob/develop/CLAUDE.md) imports it for Claude Code.
- [`.github/copilot-instructions.md`](https://github.com/roboflow/supervision/blob/develop/.github/copilot-instructions.md): the conventions GitHub Copilot follows.
- [`.github/skills/code-review/SKILL.md`](https://github.com/roboflow/supervision/blob/develop/.github/skills/code-review/SKILL.md): the checklist for reviewing a pull request or diff.

If your tool does not load these files on its own, point it at `AGENTS.md` at the start of the session. These files and the [Development guide](development.md) agree, and where they differ, the Development guide wins.

## Set up the same environment

Prepare the project yourself, exactly as for any contribution, so the assistant can run the real checks:

```bash
git clone --depth 1 -b develop https://github.com/YOUR_USERNAME/supervision.git
cd supervision
git remote add -t develop upstream https://github.com/roboflow/supervision.git
git fetch upstream
uv sync --group dev --group docs --extra metrics
```

Work on a branch from `upstream/develop`, never on `develop` itself. To run several assistants or tasks in parallel, give each its own `git worktree` so their changes do not mix.

## A workflow that works well

1. **Start from an issue or a clear problem statement.** For a new feature, discuss it in an Issue first, as for any other contribution.

2. **Ask for a plan before code.** Check that the plan fits the [API design principles](development.md#api-design-principles) and the existing `sv.Detections` and annotator patterns, and that the feature does not already exist under another name.

3. **Keep the change small.** One focused pull request is far easier to review than a large generated change. Ask the assistant to leave unrelated code alone.

4. **For a bug, reproduce it first.** Have the assistant write a failing test, then the smallest fix that makes it pass.

5. **Require tests and docs with the code.** New public functions need docstrings with an example, tests that follow the [test structure](development.md#test-structure), and a docs entry.

6. **Run the checks yourself.** Do not rely on the assistant's report that everything passes:

    ```bash
    uv run pre-commit run --all-files
    uv run pytest
    ```

7. **Read the full diff before you commit.** Remove leftovers such as unrelated edits, debugging code, invented APIs and comments that only narrate the change.

8. **Write the pull request description yourself, in your own words.** It must describe what the change does and how you tested it. Say so if an assistant wrote a substantial part, because it helps reviewers know where to look closely.

## Check the diff against the project rules

Assistants that skip the project guidance tend to break the rules below. Check for them in the diff:

- Python loops over detections where a vectorized NumPy operation belongs.
- Heavy dependencies such as `torch`, `transformers` or `ultralytics` imported at module top level instead of inside the function that needs them.
- Tests that do not follow the repository structure: several actions per test, repeated near-identical tests instead of `parametrize`, missing test docstrings.
- Docstrings that refer to local classes as plain code instead of Sphinx roles with a fully qualified `supervision.` target.
- A deprecation without both the deprecation and removal version, or a new `from_<model>` method for a model that already returns `sv.Detections`.
- Hardcoded expected values in tests that claim parity with a reference package, instead of computing the reference result with the live package.

## Using an assistant to review

An assistant is a useful first reviewer of your own change. Ask it to follow the code-review skill above and fix what it finds before you open the pull request. Treat the result as a hint, not a verdict: maintainers still review every pull request.
