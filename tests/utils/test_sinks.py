"""Round-trip tests for VideoSink and ImageSink."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from PIL import Image

import supervision as sv
import supervision.utils.video as video_utils
from supervision.utils.video import VideoInfo


def _can_create_symlink() -> bool:
    """Probe whether the current process can create filesystem symlinks."""
    with TemporaryDirectory() as directory:
        link = Path(directory) / "link"
        try:
            link.symlink_to(Path(directory) / "missing")
        except OSError:
            return False
        return link.is_symlink()


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

    def test_saves_webp_images_losslessly(self, tmp_path: Path) -> None:
        """A WebP name pattern stores the frame's exact pixels, as OpenCV does."""
        rows, columns = np.mgrid[0:48, 0:64]
        channels = (columns * 4 % 256, rows * 5 % 256, (rows + columns) * 3 % 256)
        image = np.dstack(channels).astype(np.uint8)
        with sv.ImageSink(
            target_dir_path=str(tmp_path),
            overwrite=True,
            image_name_pattern="frame_{:03d}.webp",
        ) as sink:
            sink.save_image(image=image)

        with Image.open(tmp_path / "frame_000.webp") as saved:
            saved_bgr = np.asarray(saved.convert("RGB"))[..., ::-1]
        np.testing.assert_array_equal(saved_bgr, image)

    def test_overwrite_false_reuses_existing_dir(self, tmp_path: Path) -> None:
        """Overwrite=False keeps existing directory contents intact."""
        existing = tmp_path / "existing"
        existing.mkdir()
        sentinel = existing / "keep.txt"
        sentinel.write_text("keep")

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with sv.ImageSink(target_dir_path=str(existing), overwrite=False) as sink:
            sink.save_image(image=image)

        assert sentinel.exists(), "pre-existing file should not be deleted"
        assert len(list(existing.iterdir())) == 2

    @pytest.mark.parametrize("image_name", [None, "image_00000.png"])
    def test_overwrite_false_preserves_existing_image(
        self, tmp_path: Path, image_name: str | None
    ) -> None:
        """A numbered image already on disk cannot be silently replaced."""
        existing = tmp_path / "image_00000.png"
        Image.new("RGB", (8, 8), color="black").save(existing)
        original = existing.read_bytes()
        image = np.full((8, 8, 3), 255, dtype=np.uint8)

        with sv.ImageSink(target_dir_path=str(tmp_path), overwrite=False) as sink:
            with pytest.raises(FileExistsError, match=r"image_00000\.png"):
                sink.save_image(image=image, image_name=image_name)

        assert existing.read_bytes() == original
        assert sink.image_count == 0

    def test_overwrite_true_replaces_same_named_image(self, tmp_path: Path) -> None:
        """Overwrite=True still permits replacing a named image in one session."""
        image_name = "same.png"
        black = np.zeros((8, 8, 3), dtype=np.uint8)
        white = np.full((8, 8, 3), 255, dtype=np.uint8)

        with sv.ImageSink(target_dir_path=str(tmp_path), overwrite=True) as sink:
            sink.save_image(image=black, image_name=image_name)
            sink.save_image(image=white, image_name=image_name)

        with Image.open(tmp_path / image_name) as saved:
            assert saved.getpixel((0, 0)) == (255, 255, 255)
        assert sink.image_count == 2

    @pytest.mark.parametrize("overwrite", [False, True])
    @pytest.mark.parametrize("image_name", [None, "existing.png\0suffix"])
    def test_rejects_embedded_nul_without_replacing_image(
        self, tmp_path: Path, overwrite: bool, image_name: str | None
    ) -> None:
        """Invalid generated and explicit names cannot reach native path truncation."""
        image = np.full((8, 8, 3), 255, dtype=np.uint8)
        with sv.ImageSink(
            target_dir_path=str(tmp_path),
            overwrite=overwrite,
            image_name_pattern="existing.png\0{:d}",
        ) as sink:
            existing = tmp_path / "existing.png"
            Image.new("RGB", (8, 8), color="black").save(existing)
            original = existing.read_bytes()

            with pytest.raises(ValueError, match="embedded null"):
                sink.save_image(image=image, image_name=image_name)

            assert existing.read_bytes() == original
            assert sink.image_count == 0

    @pytest.mark.skipif(
        not _can_create_symlink(), reason="Symlink creation unavailable"
    )
    def test_overwrite_false_preserves_dangling_symlink(self, tmp_path: Path) -> None:
        """A dangling link cannot create its missing target through a write."""
        target = tmp_path / "missing.png"
        link = tmp_path / "link.png"
        link.symlink_to(target.name)
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        with sv.ImageSink(target_dir_path=str(tmp_path), overwrite=False) as sink:
            with pytest.raises(FileExistsError, match=r"link\.png"):
                sink.save_image(image=image, image_name=link.name)

        assert link.is_symlink()
        assert link.readlink().name == target.name
        assert not target.exists()
        assert sink.image_count == 0

    def test_overwrite_true_clears_existing_dir(self, tmp_path: Path) -> None:
        """Overwrite=True removes pre-existing files before writing."""
        target = tmp_path / "target"
        target.mkdir()
        sentinel = target / "old.txt"
        sentinel.write_text("old")

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with sv.ImageSink(target_dir_path=str(target), overwrite=True) as sink:
            sink.save_image(image=image)

        assert not sentinel.exists(), "overwrite=True must clear pre-existing files"


class TestVideoSink:
    """VideoSink writes valid video frames to a file."""

    def test_write_frame_outside_context_raises(self, tmp_path: Path) -> None:
        """write_frame requires an open VideoSink context."""
        target = str(tmp_path / "out.avi")
        info = VideoInfo(width=64, height=64, fps=1, total_frames=None)
        sink = sv.VideoSink(target_path=target, video_info=info, codec="MJPG")

        with pytest.raises(RuntimeError, match="open VideoSink context"):
            sink.write_frame(frame=np.zeros((64, 64, 3), dtype=np.uint8))

    def test_enter_raises_when_writer_does_not_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """VideoSink surfaces VideoWriter open failures immediately."""

        class ClosedWriter:
            def isOpened(self) -> bool:
                return False

            def release(self) -> None:
                pass

        monkeypatch.setattr(
            video_utils.cv2, "VideoWriter", lambda *args: ClosedWriter()
        )
        target = str(tmp_path / "out.avi")
        info = VideoInfo(width=64, height=64, fps=1, total_frames=None)

        with pytest.raises(RuntimeError, match="Could not open video writer"):
            with sv.VideoSink(target_path=target, video_info=info, codec="MJPG"):
                pass

    def test_creates_output_file(self, tmp_path: Path) -> None:
        """VideoSink creates a non-empty file at target_path."""
        target = str(tmp_path / "out.avi")
        info = VideoInfo(width=64, height=64, fps=1, total_frames=None)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        with sv.VideoSink(target_path=target, video_info=info, codec="MJPG") as sink:
            sink.write_frame(frame=frame)

        assert Path(target).exists()
        assert Path(target).stat().st_size > 0

    def test_writes_multiple_frames(self, tmp_path: Path) -> None:
        """Writing N frames produces a strictly larger file than writing one frame."""
        info = VideoInfo(width=64, height=64, fps=5, total_frames=None)

        target_one = str(tmp_path / "one.avi")
        with sv.VideoSink(
            target_path=target_one, video_info=info, codec="MJPG"
        ) as sink:
            sink.write_frame(frame=np.zeros((64, 64, 3), dtype=np.uint8))

        target_many = str(tmp_path / "many.avi")
        with sv.VideoSink(
            target_path=target_many, video_info=info, codec="MJPG"
        ) as sink:
            for i in range(10):
                sink.write_frame(frame=np.full((64, 64, 3), i * 20, dtype=np.uint8))

        assert Path(target_many).stat().st_size > Path(target_one).stat().st_size
