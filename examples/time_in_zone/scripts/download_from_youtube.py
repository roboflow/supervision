from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Any

import yt_dlp
from yt_dlp.utils import DownloadError


def _build_ydl_opts(output_path: str | None, file_name: str | None) -> Dict[str, Any]:
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


def main(url: str, output_path: str | None, file_name: str | None) -> None:
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
    parser = argparse.ArgumentParser(
        description="Download a specific YouTube video by providing its URL."
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="The full URL of the YouTube video you wish to download.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/source",
        required=False,
        help="Optional. Specifies the directory where the video will be saved.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="video.mp4",
        required=False,
        help="Optional. Sets the name of the saved video file.",
    )
    args = parser.parse_args()
    main(url=args.url, output_path=args.output_path, file_name=args.file_name)
