import json
import os
import tempfile
import urllib.parse
from pathlib import Path
from shutil import copyfileobj
from typing import Any

import numpy as np
import requests
import yaml

SUPERVISION_CACHE_DIR = Path(tempfile.gettempdir()) / "supervision"


def _normalize_http_url(url: str) -> str:
    """
    Validate and normalize an HTTP(S) URL.

    Args:
        url: URL to validate.

    Returns:
        Normalized URL string.

    Raises:
        ValueError: If the URL is invalid or uses an unsupported scheme.
    """
    try:
        original_parsed_url = urllib.parse.urlparse(url)
        if "\\" in original_parsed_url.netloc:
            raise ValueError("URL authority contains a backslash")

        prepared_request = requests.Request(method="GET", url=url).prepare()
        prepared_url = prepared_request.url
        if prepared_url is None:
            raise ValueError("prepared URL is empty")

        parsed_url = urllib.parse.urlparse(prepared_url)
    except (requests.RequestException, ValueError) as error:
        raise ValueError(f"Invalid URL {url!r}: {error}") from error

    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError(
            f"Unsupported URL scheme {parsed_url.scheme!r} in {url!r}. "
            "Only HTTP and HTTPS URLs are supported."
        )
    if parsed_url.hostname is None:
        raise ValueError(f"Invalid URL {url!r}: no host supplied.")

    return prepared_url


def _download_to_file(
    url: str, target: Path, *, timeout: float = 30.0, stream: bool = False
) -> None:
    """
    Download `url` to `target` atomically using a temporary file and `os.replace`.

    Args:
        url: HTTP(S) URL to download.
        target: Destination file path. Parent directories are created as needed.
        timeout: Request timeout in seconds. Defaults to `30.0`.
        stream: If `True`, stream the response to disk in chunks and display a
            progress bar. If `False`, buffer the response in memory before
            writing. Defaults to `False`.

    Raises:
        requests.RequestException: If the request fails or returns an error status.
    """
    response = requests.get(url, stream=stream, allow_redirects=True, timeout=timeout)
    try:
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False, dir=target.parent, suffix=target.suffix
        ) as file:
            temp_path = Path(file.name)
            try:
                if stream:
                    from tqdm.auto import tqdm

                    file_size = int(response.headers.get("Content-Length", 0))
                    with tqdm.wrapattr(
                        response.raw, "read", total=file_size, desc="", colour="#a351fb"
                    ) as raw_response:
                        copyfileobj(raw_response, file)
                else:
                    file.write(response.content)
            except BaseException:
                file.close()
                temp_path.unlink(missing_ok=True)
                raise
    finally:
        response.close()

    try:
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


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
        extensions: A list of file extensions to filter. Extensions may be
            supplied with or without a leading dot (e.g. ``'jpg'`` and
            ``'.jpg'`` are equivalent). Matching is case-insensitive.
            Multi-part extensions are supported (e.g. ``'tar.gz'``). Pass
            ``None`` (default) to list all files; pass an empty list to
            return no files.

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
        >>> # Leading dot accepted; matching is case-insensitive
        >>> files = sv.list_files_with_extensions(
        ...     directory=tmpdir, extensions=['.txt', 'md'])
        >>> len(files)
        2

        ```
    """

    directory = Path(directory)
    files_with_extensions: list[Path] = []

    if extensions is not None:
        candidates = [p for p in directory.glob("*") if p.is_file()]
        path_index: dict[Path, set[str]] = {}
        for path in candidates:
            suffixes = [suffix.lower().lstrip(".") for suffix in path.suffixes]
            path_index[path] = {
                ".".join(suffixes[index:]) for index in range(len(suffixes))
            }
        seen_paths: set[Path] = set()
        for ext in extensions:
            normalized_extension = ext.lower().lstrip(".")
            if not normalized_extension:
                continue
            for path, path_extensions in path_index.items():
                if path not in seen_paths and normalized_extension in path_extensions:
                    files_with_extensions.append(path)
                    seen_paths.add(path)
    else:
        files_with_extensions.extend(p for p in directory.glob("*") if p.is_file())

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
