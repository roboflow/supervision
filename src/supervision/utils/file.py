from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def list_files_with_extensions(
    directory: str | Path, extensions: list[str] | None = None
) -> list[Path]:
    """
    List files in a directory with specified extensions or
        all files if no extensions are provided.

    Args:
        directory: The directory path as a string or Path object.
        extensions: A list of file extensions to filter.
            Default is None, which lists all files.

    Returns:
        A list of Path objects for the matching files.

    Examples:
        ```pycon
        >>> import supervision as sv
        >>> from pathlib import Path
        >>> import tempfile
        >>> # Keep a reference to the directory object
        >>> tmp_dir_obj = tempfile.TemporaryDirectory()
        >>> tmpdir = tmp_dir_obj.name
        >>> # Create test files
        >>> (Path(tmpdir) / "test1.txt").touch()
        >>> (Path(tmpdir) / "test2.md").touch()
        >>> (Path(tmpdir) / "test3.py").touch()
        >>> # List all files in the directory
        >>> files = sv.list_files_with_extensions(directory=tmpdir)
        >>> len(files)
        3
        >>> # List only files with '.txt' and '.md' extensions
        >>> files = sv.list_files_with_extensions(
        ...     directory=tmpdir, extensions=['txt', 'md'])
        >>> len(files)
        2

        ```
    """

    directory = Path(directory)
    files_with_extensions: list[Path] = []

    if extensions is not None:
        for ext in extensions:
            files_with_extensions.extend(directory.glob(f"*.{ext}"))
    else:
        files_with_extensions.extend(directory.glob("*"))

    return files_with_extensions


def read_txt_file(file_path: str | Path, skip_empty: bool = False) -> list[str]:
    """
    Read a text file and return a list of strings without newline characters.
    Optionally skip empty lines.

    Args:
        file_path: The file path as a string or Path object.
        skip_empty: If True, skip lines that are empty or contain only
            whitespace. Default is False.

    Returns:
        A list of strings representing the lines in the text file.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from supervision.utils.file import read_txt_file, save_text_file
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     file_path = Path(tmpdir) / "test.txt"
        ...     save_text_file(["line1", " ", "line3"], file_path)
        ...     print(read_txt_file(file_path))
        ...     print(read_txt_file(file_path, skip_empty=True))
        ['line1', ' ', 'line3']
        ['line1', 'line3']

        ```
    """
    with open(str(file_path)) as file:
        if skip_empty:
            lines = [line.rstrip("\n") for line in file if line.strip()]
        else:
            lines = [line.rstrip("\n") for line in file]

    return lines


def save_text_file(lines: list[str], file_path: str | Path) -> None:
    """
    Write a list of strings to a text file, each string on a new line.

    Args:
        lines: The list of strings to be written to the file.
        file_path: The file path as a string or Path object.
    """
    with open(str(file_path), "w") as file:
        for line in lines:
            file.write(line + "\n")


def read_json_file(file_path: str | Path) -> dict[str, Any]:
    """
    Read a json file and return a dict.

    Args:
        file_path: The file path as a string or Path object.

    Returns:
        A dict of annotations information

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from supervision.utils.file import read_json_file, save_json_file
        >>> data = {"key": "value", "list": [1, 2, 3]}
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     file_path = Path(tmpdir) / "test.json"
        ...     save_json_file(data, file_path)
        ...     print(read_json_file(file_path))
        {'key': 'value', 'list': [1, 2, 3]}

        ```
    """
    with open(str(file_path)) as file:
        data = json.load(file)
    return data  # type: ignore


def save_json_file(
    data: dict[str, Any], file_path: str | Path, indent: int = 3
) -> None:
    """
    Write a dict to a json file.

    Args:
        data: dict with unique keys and value as pair.
        file_path: The file path as a string or Path object.
        indent:
    """
    with open(str(file_path), "w") as fp:
        json.dump(data, fp, cls=NumpyJsonEncoder, indent=indent)


def read_yaml_file(file_path: str | Path) -> dict[str, Any]:
    """
    Read a yaml file and return a dict.

    Args:
        file_path: The file path as a string or Path object.

    Returns:
        A dict of content information
    """
    with open(str(file_path)) as file:
        data = yaml.safe_load(file)
    return data  # type: ignore


def save_yaml_file(data: dict[str, Any], file_path: str | Path) -> None:
    """
    Save a dict to a yaml file.

    Args:
        data: dict with unique keys and value as pair.
        file_path: The file path as a string or Path object.
    """

    with open(str(file_path), "w") as outfile:
        yaml.dump(data, outfile, sort_keys=False, default_flow_style=None)
