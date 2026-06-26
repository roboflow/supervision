from __future__ import annotations

import os
import sys
from typing import Any

import yt_dlp
from jsonargparse import auto_cli
from yt_dlp.utils import DownloadError


def _build_ydl_opts(output_path: str | None, file_name: str | None) -> dict[str, Any]:
    out_dir = output_path or "."

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    name_template = file_name if file_name else "%(title)s.%(ext)s"

    return {
        "format": (
            "bestvideo[ext=mp4][vcodec!*=av01][height<=2160]+bestaudio[ext=m4a]/"
            "best[ext=mp4][vcodec!*=av01][height<=2160]/"
            "bestvideo+bestaudio/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(out_dir, name_template),
        "quiet": False,
        "noplaylist": True,
    }


def main(
    url: str, output_path: str = "data/source", file_name: str = "video.mp4"
) -> None:
    """
    Download a specific YouTube video by providing its URL.

    Args:
        url: The full URL of the YouTube video you wish to download.
        output_path: Specifies the directory where the video will be saved.
        file_name: Sets the name of the saved video file.
    """
    # ssl._create_default_https_context = ssl._create_unverified_context
    ydl_opts = _build_ydl_opts(output_path, file_name)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except DownloadError as err:
        print(f"Download failed: {err}", file=sys.stderr)
        sys.exit(1)

    final_name = file_name if file_name else "the video title"
    final_path = output_path if output_path else "current directory"
    print(f"Download completed! Video saved as '{final_name}' in '{final_path}'.")


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
