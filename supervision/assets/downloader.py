import os
from functools import partial
from hashlib import md5
from os import remove as os_remove
from os.path import exists as path_exists
from pathlib import Path
from shutil import copyfileobj
from typing import Union

from supervision.assets import VideoAssets
from supervision.assets.asset_list import VIDEO_ASSETS

try:
    from requests import Response, get
    from tqdm.auto import tqdm
except ImportError:
    raise ValueError(
        "\n"
        "Please install requests and tqdm to download assets \n"
        "or install supervision with assets \n"
        "pip install supervision[assets] \n"
        "\n"
    )


def is_md5_hash_matching(filename: str, original_md5_hash: str) -> bool:
    """
    Check if the MD5 hash of a file matches the original hash.

    Parameters:
        filename (str): The path to the file to be checked as a string.
        original_md5_hash (str): The original MD5 hash to compare against.

    Returns:
        bool: True if the hashes match, False otherwise.
    """
    if not os.path.exists(filename):
        return False

    with open(filename, "rb") as file:
        file_contents = file.read()
        computed_md5_hash = md5(file_contents).hexdigest()

    return computed_md5_hash == original_md5_hash


def download_assets(asset_name: Union[VideoAssets, str]) -> str:
    """
    asset_name: VIDEO_ASSETS,  name of the file to download
    """

    filename = asset_name.value if isinstance(asset_name, VideoAssets) else asset_name
    if not path_exists(filename) and filename in VIDEO_ASSETS:
        print(f"Downloading {filename} assets \n")
        res: Response = get(
            VIDEO_ASSETS[filename][0], stream=True, allow_redirects=True
        )
        if res.status_code != 200:
            res.raise_for_status()
            raise RuntimeError(
                f"Request to {VIDEO_ASSETS[asset_name][0]} "
                f"returned status code {res.status_code}"
            )

        file_size: int = int(res.headers.get("Content-Length", 0))
        folder_path: Path = Path(filename).expanduser().resolve()
        folder_path.parent.mkdir(parents=True, exist_ok=True)

        desc = "(Unknown total file size)" if file_size == 0 else ""
        res.raw.read = partial(res.raw.read, decode_content=True)
        with tqdm.wrapattr(
            res.raw,
            "read",
            total=file_size,
            desc=desc,
            colour="#a351fb",
        ) as r_raw:
            with folder_path.open("wb") as f:
                copyfileobj(r_raw, f)

    elif path_exists(filename):
        if not is_md5_hash_matching(filename, VIDEO_ASSETS[filename][2]):
            print("File corrupted. Re-downloading... \n")
            os_remove(filename)
            download_assets(filename)

        print(f"{filename} asset download complete. \n")
    else:
        raise ValueError(
            f"Invalid asset. It should be one of the following: {VideoAssets.list()}."
        )

    return filename
