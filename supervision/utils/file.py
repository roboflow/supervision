from pathlib import Path
from typing import List, Optional, Union


def list_files_with_extensions(
    directory: Union[str, Path], extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    List files in a directory with specified extensions or all files if no extensions are provided.

    Args:
        directory (Union[str, Path]): The directory path as a string or Path object.
        extensions (Optional[List[str]]): A list of file extensions to filter. Default is None, which lists all files.

    Returns:
        (List[Path]): A list of Path objects for the matching files.

    Examples:
        ```python
        >>> import supervision as sv

        >>> # List all files in the directory
        >>> files = sv.list_files_with_extensions(directory='my_directory')

        >>> # List only files with '.txt' and '.md' extensions
        >>> files = sv.list_files_with_extensions(directory='my_directory', extensions=['txt', 'md'])
        ```
    """
    directory = Path(directory)
    files_with_extensions = []

    if extensions is not None:
        for ext in extensions:
            files_with_extensions.extend(directory.glob(f"*.{ext}"))
    else:
        files_with_extensions.extend(directory.glob("*"))

    return files_with_extensions


def read_txt_file(file_path: str) -> List[str]:
    """
    Read a text file and return a list of strings without newline characters.

    Args:
        file_path (str): The path to the text file.

    Returns:
        List[str]: A list of strings representing the lines in the text file.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
        lines = [line.rstrip("\n") for line in lines]

    return lines


def save_text_file(lines: List[str], file_path: str):
    """
    Write a list of strings to a text file, each string on a new line.

    Args:
        lines (List[str]): The list of strings to be written to the file.
        file_path (str): The path to the text file.
    """
    with open(file_path, "w") as file:
        for line in lines:
            file.write(line + "\n")
