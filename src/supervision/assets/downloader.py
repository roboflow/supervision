import os
from hashlib import md5
from pathlib import Path
from shutil import copyfileobj

from requests import get
from tqdm.auto import tqdm

from supervision.assets.list import MEDIA_ASSETS, Assets
from supervision.utils.logger import _get_logger

logger = _get_logger(__name__)


def is_md5_hash_matching(filename: str, original_md5_hash: str) -> bool:
    """
    Check if the MD5 hash of a file matches the original hash.

    Note: MD5 is used here for file integrity checking (detecting corruption),
    not for cryptographic security purposes.

    Args:
        filename: The path to the file to be checked.
        original_md5_hash: The original MD5 hash to compare against.

    Returns:
        True if the hashes match, False otherwise.
    """
    if not os.path.exists(filename):
        return False

    with open(filename, "rb") as file:
        file_contents = file.read()
        computed_md5_hash = md5(file_contents, usedforsecurity=False)

    return computed_md5_hash.hexdigest() == original_md5_hash


def _download_asset(filename: str) -> None:
    """
    Download asset bytes to the target filename.
    """
    response = get(
        MEDIA_ASSETS[filename][0], stream=True, allow_redirects=True, timeout=30
    )
    response.raise_for_status()

    file_size = int(response.headers.get("Content-Length", 0))
    folder_path = Path(filename).expanduser().resolve()
    folder_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm.wrapattr(
        response.raw, "read", total=file_size, desc="", colour="#a351fb"
    ) as raw_resp:
        with folder_path.open("wb") as file:
            copyfileobj(raw_resp, file)


def _download_verified_asset(
    filename: str, original_md5_hash: str, retry_on_mismatch: bool = True
) -> None:
    """
    Download an asset and reject payloads whose MD5 does not match the catalog.
    """
    _download_asset(filename)

    if is_md5_hash_matching(filename, original_md5_hash):
        return

    logger.warning("File corrupted. Re-downloading...")
    os.remove(filename)

    if retry_on_mismatch:
        _download_verified_asset(
            filename=filename,
            original_md5_hash=original_md5_hash,
            retry_on_mismatch=False,
        )
        return

    raise ValueError(f"Downloaded asset {filename!r} failed MD5 verification.")


def download_assets(asset_name: Assets | str) -> str:
    """
    Download a specified asset if it doesn't already exist or is corrupted.

    Args:
        asset_name: The name or type of the asset to be downloaded.

    Returns:
        The filename of the downloaded asset.

    Example:
        ```pycon
        >>> from supervision.assets import download_assets, ImageAssets, VideoAssets
        >>> download_assets(VideoAssets.VEHICLES)  # doctest: +SKIP
        'vehicles.mp4'

        >>> download_assets(ImageAssets.PEOPLE_WALKING)  # doctest: +SKIP
        'people-walking.jpg'

        ```
    """

    filename = asset_name.filename if isinstance(asset_name, Assets) else asset_name

    if filename in MEDIA_ASSETS:
        original_md5_hash = MEDIA_ASSETS[filename][1]
        if not Path(filename).exists():
            logger.info("Downloading %s assets", filename)
            _download_verified_asset(filename, original_md5_hash)
        else:
            if not is_md5_hash_matching(filename, original_md5_hash):
                logger.warning("File corrupted. Re-downloading...")
                os.remove(filename)
                _download_verified_asset(filename, original_md5_hash)

            logger.info("%s asset download complete.", filename)
    else:
        valid_assets = ", ".join(filename for filename in MEDIA_ASSETS.keys())
        raise ValueError(
            f"Invalid asset. It should be one of the following: {valid_assets}."
        )

    return filename
