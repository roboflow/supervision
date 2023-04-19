from pathlib import Path
from typing import List, Optional, Union


def write_text_to_file(content: str, output_path: str) -> None:
    with open(output_path, "w") as file:
        file.write(content)


def list_files_with_extensions(
    directory: Union[str, Path], extensions: Optional[List[str]] = None
) -> List[Path]:
    directory = Path(directory)
    files_with_extensions = []

    if extensions is not None:
        for ext in extensions:
            files_with_extensions.extend(directory.glob(f"*.{ext}"))
    else:
        files_with_extensions.extend(directory.glob("*"))

    return files_with_extensions
