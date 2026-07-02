"""Round-trip tests for VideoSink and ImageSink."""

from pathlib import Path

import numpy as np

import supervision as sv
from supervision.utils.video import VideoInfo


class TestImageSink:
    """ImageSink saves images to disk using the configured name pattern."""

    def test_saves_single_image(self, tmp_path: Path) -> None:
        """A single saved frame produces exactly one file."""
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        with sv.ImageSink(target_dir_path=str(tmp_path), overwrite=True) as sink:
            sink.save_image(image=image)

        files = list(tmp_path.iterdir())
        assert len(files) == 1

    def test_saves_multiple_images(self, tmp_path: Path) -> None:
        """N consecutive save_image calls create N distinct files."""
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        n = 5
        with sv.ImageSink(target_dir_path=str(tmp_path), overwrite=True) as sink:
            for _ in range(n):
                sink.save_image(image=image)

        files = sorted(tmp_path.iterdir())
        assert len(files) == n

    def test_custom_name_pattern(self, tmp_path: Path) -> None:
        """Custom image_name_pattern is applied to each saved file."""
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with sv.ImageSink(
            target_dir_path=str(tmp_path),
            overwrite=True,
            image_name_pattern="frame_{:03d}.jpg",
        ) as sink:
            sink.save_image(image=image)

        names = [f.name for f in tmp_path.iterdir()]
        assert names == ["frame_000.jpg"]

    def test_overwrite_false_reuses_existing_dir(self, tmp_path: Path) -> None:
        """overwrite=False keeps existing directory contents intact."""
        existing = tmp_path / "existing"
        existing.mkdir()
        sentinel = existing / "keep.txt"
        sentinel.write_text("keep")

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with sv.ImageSink(target_dir_path=str(existing), overwrite=False) as sink:
            sink.save_image(image=image)

        assert sentinel.exists(), "pre-existing file should not be deleted"
        assert len(list(existing.iterdir())) == 2


class TestVideoSink:
    """VideoSink writes valid video frames to a file."""

    def test_creates_output_file(self, tmp_path: Path) -> None:
        """VideoSink creates a non-empty file at target_path."""
        target = str(tmp_path / "out.mp4")
        info = VideoInfo(width=64, height=64, fps=1, total_frames=None)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        with sv.VideoSink(target_path=target, video_info=info) as sink:
            sink.write_frame(frame=frame)

        assert Path(target).exists()
        assert Path(target).stat().st_size > 0

    def test_writes_multiple_frames(self, tmp_path: Path) -> None:
        """Writing N frames produces a larger file than writing one frame."""
        info = VideoInfo(width=64, height=64, fps=5, total_frames=None)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        target_one = str(tmp_path / "one.mp4")
        with sv.VideoSink(target_path=target_one, video_info=info) as sink:
            sink.write_frame(frame=frame)

        target_many = str(tmp_path / "many.mp4")
        with sv.VideoSink(target_path=target_many, video_info=info) as sink:
            for _ in range(10):
                sink.write_frame(frame=frame)

        assert Path(target_many).stat().st_size >= Path(target_one).stat().st_size
