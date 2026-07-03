import subprocess
from pathlib import Path

import imageio_ffmpeg


def ensure_browser_playable(video_path: Path) -> Path:
    """Transcode a video to H.264 so browsers can play it in HTML5 video.

    OpenCV VideoSink defaults to ``mp4v``, which most browsers cannot decode.
    This re-encodes the file in place using the bundled ffmpeg from imageio-ffmpeg.

    Args:
        video_path: Path to the video file to transcode.

    Returns:
        The same path, now containing a browser-compatible MP4.

    Examples:
        >>> ensure_browser_playable(Path("output.mp4"))  # doctest: +SKIP
        PosixPath('output.mp4')
    """
    temp_path = video_path.with_name(f"{video_path.stem}_web{video_path.suffix}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(temp_path),
        ],
        check=True,
        capture_output=True,
    )

    video_path.unlink(missing_ok=True)
    temp_path.rename(video_path)
    return video_path
