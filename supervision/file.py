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
        >>> from supervision import list_files_with_extensions

        >>> # List all files in the directory
        >>> files = list_files_with_extensions(directory='my_directory')

        >>> # List only files with '.txt' and '.md' extensions
        >>> files = list_files_with_extensions(directory='my_directory', extensions=['txt', 'md'])
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
