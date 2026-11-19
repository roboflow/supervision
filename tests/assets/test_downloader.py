import io
import json
import os
import subprocess
import sys
from hashlib import md5
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from supervision.assets.downloader import (
    _download_asset,
    download_assets,
    is_md5_hash_matching,
)
from supervision.assets.list import MEDIA_ASSETS, ImageAssets, VideoAssets


def _mock_streaming_response(payload: bytes = b"asset-bytes") -> MagicMock:
    response = MagicMock()
    response.headers = {"Content-Length": str(len(payload))}
    response.raw = io.BytesIO(payload)
    response.raise_for_status = MagicMock()
    return response


class TestMD5HashMatching:
    def test_uses_bounded_memory(self, tmp_path: Path) -> None:
        """Avoid retaining the entire asset, even when reads are chunked."""
        content = b"a" * (8 * 1024 * 1024) + b"tail"
        filename = tmp_path / "asset.bin"
        filename.write_bytes(content)
        expected = md5(content, usedforsecurity=False).hexdigest()
        environment = os.environ.copy()
        environment.pop("PYTHONTRACEMALLOC", None)
        script = """
import json
import sys
import tracemalloc
from supervision.assets.downloader import is_md5_hash_matching

tracemalloc.start()
matches = is_md5_hash_matching(sys.argv[1], sys.argv[2])
_, peak_bytes = tracemalloc.get_traced_memory()
print(json.dumps({"matches": matches, "peak_bytes": peak_bytes}))
"""

        completed = subprocess.run(
            [sys.executable, "-c", script, str(filename), expected],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )

        measurement = json.loads(completed.stdout)
        assert measurement["matches"]
        # A whole-file buffer or a retained list of chunks exceeds this bound.
        assert measurement["peak_bytes"] < len(content) // 2

    @pytest.mark.parametrize("size", [0, 1, 65535, 1048576, 3145765])
    @pytest.mark.parametrize("as_string", [False, True])
    def test_matches_reference_for_complete_file(
        self, tmp_path: Path, size: int, as_string: bool
    ) -> None:
        """Match hashlib for empty, small, and multi-chunk files with a tail."""
        content = (bytes(range(256)) * ((size + 255) // 256))[:size]
        filename = tmp_path / "asset.bin"
        filename.write_bytes(content)
        expected = md5(content, usedforsecurity=False).hexdigest()
        target = str(filename) if as_string else filename

        matches = is_md5_hash_matching(target, expected)

        assert matches

    def test_rejects_changed_tail(self, tmp_path: Path) -> None:
        """Include the final byte when checking a large asset for corruption."""
        content = b"a" * (3 * 1024 * 1024) + b"tail"
        filename = tmp_path / "asset.bin"
        filename.write_bytes(content[:-1] + b"!")
        expected = md5(content, usedforsecurity=False).hexdigest()

        matches = is_md5_hash_matching(filename, expected)

        assert not matches

    def test_keeps_case_sensitive_comparison(self, tmp_path: Path) -> None:
        """Preserve the catalog hash's existing case-sensitive comparison."""
        content = b"asset bytes"
        filename = tmp_path / "asset.bin"
        filename.write_bytes(content)
        expected = md5(content, usedforsecurity=False).hexdigest().upper()

        matches = is_md5_hash_matching(filename, expected)

        assert not matches

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(OSError("hash read failed"), id="read-error"),
            pytest.param(KeyboardInterrupt(), id="cancelled-read"),
        ],
    )
    def test_closes_file_when_later_read_fails(self, error: BaseException) -> None:
        """Propagate read failures and cancellation while closing the input."""
        reader = MagicMock(spec=io.BufferedReader)
        reader.__enter__.return_value = reader
        reader.read.side_effect = [b"first chunk", error]

        with (
            patch("builtins.open", return_value=reader),
            patch("os.path.exists", return_value=True),
            pytest.raises(type(error)),
        ):
            is_md5_hash_matching("asset.bin", "wrong hash")

        reader.__exit__.assert_called_once()

    def test_file_exists_matching_hash(self) -> None:
        """Test is_md5_hash_matching when file exists and hash matches."""
        test_content = b"test content"
        test_hash = "9473fdd0d880a43c21b7778d34872157"  # MD5 of "test content"

        with (
            patch("builtins.open", mock_open(read_data=test_content)),
            patch("os.path.exists", return_value=True),
        ):
            assert is_md5_hash_matching("dummy_file", test_hash)

    def test_file_exists_not_matching_hash(self) -> None:
        """Test is_md5_hash_matching when file exists but hash doesn't match."""
        test_content = b"test content"
        wrong_hash = "wrong_hash"

        with (
            patch("builtins.open", mock_open(read_data=test_content)),
            patch("os.path.exists", return_value=True),
        ):
            assert not is_md5_hash_matching("dummy_file", wrong_hash)

    def test_file_not_exists(self) -> None:
        """Test is_md5_hash_matching when file doesn't exist."""
        with patch("os.path.exists", return_value=False):
            assert not is_md5_hash_matching("nonexistent_file", "some_hash")


class TestDownloadAssets:
    @pytest.mark.parametrize("custom_directory", [False, True])
    def test_valid_cache_uses_bounded_reads_without_downloading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        custom_directory: bool,
    ) -> None:
        """Validate a cached asset without reading the whole file or networking."""
        monkeypatch.chdir(tmp_path)
        filename = VideoAssets.VEHICLES.filename
        content = b"asset frame" * (800000) + b"tail"
        target = tmp_path / filename
        target.write_bytes(content)
        expected_hash = md5(content, usedforsecurity=False).hexdigest()
        monkeypatch.setitem(
            MEDIA_ASSETS, filename, ("https://example.com", expected_hash)
        )
        directory = tmp_path if custom_directory else None
        expected_result = str(target) if custom_directory else filename
        read_sizes: list[int] = []

        with target.open("rb") as stream:
            reader = MagicMock(wraps=stream)
            reader.__enter__.return_value = reader
            reader.__exit__.side_effect = stream.__exit__

            def recording_read(size: int = -1) -> bytes:
                """Record requested sizes while reading the real cached file."""
                read_sizes.append(size)
                return stream.read(size)

            reader.read.side_effect = recording_read
            with (
                patch(
                    "supervision.assets.downloader.open",
                    return_value=reader,
                    create=True,
                ),
                patch("supervision.utils.file.requests.get") as network,
            ):
                result = download_assets(VideoAssets.VEHICLES, directory=directory)

            assert result == expected_result
            assert read_sizes
            assert all(0 < size < len(content) // 2 for size in read_sizes)
            assert stream.closed
            network.assert_not_called()

    @patch("supervision.assets.downloader.logger")
    @patch("supervision.assets.downloader.is_md5_hash_matching", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_already_exists_and_valid(
        self, mock_exists: MagicMock, mock_md5: MagicMock, mock_logger: MagicMock
    ) -> None:
        """Test download_assets when file already exists and is valid."""
        filename = "vehicles.mp4"
        result = download_assets(filename)
        assert result == filename
        mock_logger.info.assert_called_with("%s asset download complete.", filename)

    @patch("supervision.assets.downloader.logger")
    @patch("os.remove")
    @patch("supervision.assets.downloader._download_asset")
    @patch(
        "supervision.assets.downloader.is_md5_hash_matching",
        side_effect=[False, True],
    )
    @patch("pathlib.Path.exists", return_value=True)
    def test_already_exists_but_corrupted(
        self,
        mock_exists: MagicMock,
        mock_md5: MagicMock,
        mock_download: MagicMock,
        mock_remove: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Test download_assets when file exists but is corrupted (re-downloads)."""
        filename = "vehicles.mp4"

        result = download_assets(filename)

        assert result == filename
        mock_download.assert_called_once()
        mock_logger.warning.assert_called_once_with("File corrupted. Re-downloading...")
        mock_remove.assert_called_once_with(filename)

    @patch("supervision.assets.downloader.logger")
    @patch("supervision.assets.downloader._download_asset")
    @patch("supervision.assets.downloader.is_md5_hash_matching", return_value=True)
    @patch("pathlib.Path.exists", return_value=False)
    def test_download_new_file(
        self,
        mock_exists: MagicMock,
        mock_md5: MagicMock,
        mock_download: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Test download_assets verifies a freshly downloaded file."""
        filename = "vehicles.mp4"

        result = download_assets(filename)

        assert result == filename
        mock_logger.info.assert_called_with("Downloading %s assets", filename)
        mock_download.assert_called_once_with(filename, Path.cwd() / filename)
        mock_md5.assert_called_once_with(filename, "8155ff4e4de08cfa25f39de96483f918")

    @patch("supervision.assets.downloader.logger")
    @patch("os.remove")
    @patch("supervision.assets.downloader._download_asset")
    @patch(
        "supervision.assets.downloader.is_md5_hash_matching",
        side_effect=[False, True],
    )
    @patch("pathlib.Path.exists", return_value=False)
    def test_download_new_file_retries_corrupted_payload(
        self,
        mock_exists: MagicMock,
        mock_md5: MagicMock,
        mock_download: MagicMock,
        mock_remove: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Test download_assets retries once when a fresh payload fails MD5."""
        filename = "vehicles.mp4"

        result = download_assets(filename)

        assert result == filename
        assert mock_download.call_count == 2
        mock_remove.assert_called_once_with(filename)
        mock_logger.warning.assert_called_once_with("File corrupted. Re-downloading...")

    @patch("supervision.assets.downloader.logger")
    @patch("os.remove")
    @patch("supervision.assets.downloader._download_asset")
    @patch(
        "supervision.assets.downloader.is_md5_hash_matching",
        side_effect=[False, False],
    )
    @patch("pathlib.Path.exists", return_value=False)
    def test_download_new_file_raises_after_second_md5_mismatch(
        self,
        mock_exists: MagicMock,
        mock_md5: MagicMock,
        mock_download: MagicMock,
        mock_remove: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Test download_assets fails after the verified retry is also corrupted."""
        filename = "vehicles.mp4"

        with pytest.raises(ValueError, match="failed MD5 verification"):
            download_assets(filename)

        assert mock_download.call_count == 2
        assert mock_remove.call_count == 2
        assert mock_logger.warning.call_count == 2

    @patch("supervision.assets.downloader.logger")
    @patch("supervision.assets.downloader.is_md5_hash_matching", return_value=True)
    def test_download_new_file_to_custom_directory(
        self, mock_md5: MagicMock, mock_logger: MagicMock, tmp_path: Path
    ) -> None:
        """Test download_assets writes into an explicit output directory."""
        filename = "vehicles.mp4"
        target_directory = tmp_path / "nested" / "assets"
        response = _mock_streaming_response(b"asset-bytes")

        with patch("supervision.utils.file.requests.get", return_value=response):
            result = download_assets(filename, directory=target_directory)

        assert result == str(target_directory / filename)
        assert (target_directory / filename).exists()
        assert (target_directory / filename).read_bytes() == b"asset-bytes"
        mock_md5.assert_called_once_with(
            str(target_directory / filename), "8155ff4e4de08cfa25f39de96483f918"
        )

    @patch("os.replace")
    def test_partial_download_does_not_leave_final_file(
        self, mock_replace: MagicMock, tmp_path: Path
    ) -> None:
        """Test _download_asset stages downloads so failed replaces do not leak."""
        filename = "vehicles.mp4"
        destination = tmp_path / filename
        response = _mock_streaming_response(b"partial")
        mock_replace.side_effect = OSError("boom")

        with (
            patch("supervision.utils.file.requests.get", return_value=response),
            pytest.raises(OSError, match="boom"),
        ):
            _download_asset(filename, destination)

        assert not destination.exists()
        assert list(tmp_path.iterdir()) == []

    @patch("pathlib.Path.exists", return_value=False)
    def test_invalid_asset(self, mock_exists: MagicMock) -> None:
        """Test download_assets with invalid asset name."""
        invalid_filename = "invalid.mp4"

        with pytest.raises(ValueError, match="Invalid asset") as exc_info:
            download_assets(invalid_filename)

        assert "Invalid asset" in str(exc_info.value)
        assert "vehicles.mp4" in str(exc_info.value)

    @patch("pathlib.Path.exists", return_value=True)
    def test_invalid_asset_when_file_exists(self, mock_exists: MagicMock) -> None:
        """Test download_assets with invalid asset name that already exists."""
        invalid_filename = "invalid.mp4"

        with pytest.raises(ValueError, match="Invalid asset") as exc_info:
            download_assets(invalid_filename)

        assert "Invalid asset" in str(exc_info.value)
        assert "vehicles.mp4" in str(exc_info.value)

    @patch("supervision.assets.downloader.logger")
    @patch("supervision.assets.downloader._download_asset")
    @patch("supervision.assets.downloader.is_md5_hash_matching", return_value=True)
    @patch("pathlib.Path.exists", return_value=False)
    def test_with_video_enum(
        self,
        mock_exists: MagicMock,
        mock_md5: MagicMock,
        mock_download: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Test download_assets with VideoAssets enum."""
        asset = VideoAssets.VEHICLES

        result = download_assets(asset)

        assert result == asset.filename
        mock_logger.info.assert_called_with("Downloading %s assets", asset.filename)
        mock_download.assert_called_once_with(
            asset.filename, Path.cwd() / asset.filename
        )
        mock_md5.assert_called_once_with(
            asset.filename, MEDIA_ASSETS[asset.filename][1]
        )

    @patch("supervision.assets.downloader.logger")
    @patch("supervision.assets.downloader._download_asset")
    @patch("supervision.assets.downloader.is_md5_hash_matching", return_value=True)
    @patch("pathlib.Path.exists", return_value=False)
    def test_with_image_enum(
        self,
        mock_exists: MagicMock,
        mock_md5: MagicMock,
        mock_download: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Test download_assets with ImageAssets enum."""
        asset = ImageAssets.SOCCER

        result = download_assets(asset)

        assert result == asset.filename
        mock_logger.info.assert_called_with("Downloading %s assets", asset.filename)
        mock_download.assert_called_once_with(
            asset.filename, Path.cwd() / asset.filename
        )
        mock_md5.assert_called_once_with(
            asset.filename, MEDIA_ASSETS[asset.filename][1]
        )
