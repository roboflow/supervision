from __future__ import annotations

import os
import ssl

from jsonargparse import auto_cli
from pytubefix import YouTube


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
    ssl._create_default_https_context = ssl._create_unverified_context

    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    stream.download(output_path=output_path, filename=file_name)
    final_name = file_name if file_name else yt.title
    final_path = output_path if output_path else "current directory"
    print(f"Download completed! Video saved as '{final_name}' in '{final_path}'.")


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
