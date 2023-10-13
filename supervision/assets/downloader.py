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


# md5 check function
def _check_md5(filename: str, orig_md5: str) -> bool:
    """
    filename: str, A string representing the path to the file.
    orig_md5: str, A string representing the original md5 hash.
    """
    if not path_exists(filename):
        return False
    with open(filename, "rb") as file_to_check:
        # read contents of the file
        data = file_to_check.read()
        # pipe contents of the file through
        md5_returned = md5(data).hexdigest()
        # Return True if the computed hash matches the original one
        if md5_returned == orig_md5:
            return True
        return False


def download_assets(asset_name: Union[VideoAssets, str]) -> str:
    """
    asset_name: VIDEO_ASSETS,  name of the file to download
    """

    filename = f"{asset_name}"
    if not path_exists(filename) and asset_name in VIDEO_ASSETS:
        print(f"Downloading {filename} assets \n")
        res: Response = get(
            VIDEO_ASSETS[asset_name][0], stream=True, allow_redirects=True
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
        res.raw.read = partial(
            res.raw.read, decode_content=True
        )  # Decompress if needed
        with tqdm.wrapattr(
            res.raw,
            "read",
            total=file_size,
            desc=desc,
            colour="a351fb",
        ) as r_raw:
            with folder_path.open("wb") as f:
                copyfileobj(r_raw, f)

    elif path_exists(filename):
        if not _check_md5(filename, VIDEO_ASSETS[asset_name][2]):
            print("File corrupted. Re-downloading... \n")
            os_remove(filename)
            download_assets(asset_name)

        print(f"{filename} asset download complete. \n")
    else:
        raise ValueError(
            "Invalid asset type. It should be one of the following: \n"
            "vehicles.mp4 \n"
            "vehicles-2.mp4 \n"
            "milk-video-1.mp4 \n"
            "grocery-store.mp4 \n"
            "subway.mp4 \n"
            "market-square.mp4 \n"
            "people-walking-bw.mp4 \n"
        )

    return filename
