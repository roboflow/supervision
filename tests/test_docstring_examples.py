"""Guard that the fenced Python examples in docstrings are valid Python."""

import ast
import re
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

import supervision as sv

PACKAGE_ROOT = Path(sv.__file__).parent
PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

Definition = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


def _iter_definitions(node: ast.AST, qualname: str) -> Iterator[tuple[str, Definition]]:
    """Yield every class and function under ``node`` with its qualified name."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            child_qualname = f"{qualname}.{child.name}" if qualname else child.name
            yield child_qualname, child
            yield from _iter_definitions(child, child_qualname)
        else:
            yield from _iter_definitions(child, qualname)


def _collect_python_examples() -> list[tuple[str, str]]:
    """Return ``(id, code)`` for every fenced ``python`` block in a docstring.

    These blocks are not run as doctests, so nothing else checks that they parse.
    """
    examples = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owners: list[tuple[str, ast.Module | Definition]] = [
            ("<module>", tree),
            *_iter_definitions(tree, ""),
        ]
        for qualname, node in owners:
            docstring = ast.get_docstring(node)
            if not docstring:
                continue
            for index, match in enumerate(PYTHON_FENCE.finditer(docstring)):
                example_id = (
                    f"{path.relative_to(PACKAGE_ROOT).as_posix()}:{qualname}:{index}"
                )
                examples.append((example_id, textwrap.dedent(match.group(1))))
    return examples


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(code, id=example_id)
        for example_id, code in _collect_python_examples()
    ],
)
def test_docstring_python_example_parses(example: str) -> None:
    """A fenced ``python`` example in a docstring must be valid Python syntax."""
    ast.parse(example)
