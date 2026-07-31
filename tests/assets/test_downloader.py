import io
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
    @patch("supervision.assets.downloader.logger")
    @patch("supervision.assets.downloader.is_md5_hash_matching", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_already_exists_and_valid(self, mock_exists, mock_md5, mock_logger) -> None:
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
        self, mock_exists, mock_md5, mock_download, mock_remove, mock_logger
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
        self, mock_exists, mock_md5, mock_download, mock_logger
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
        self, mock_exists, mock_md5, mock_download, mock_remove, mock_logger
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
        self, mock_exists, mock_md5, mock_download, mock_remove, mock_logger
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
        self, mock_md5, mock_logger, tmp_path
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
        self, mock_replace, tmp_path
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
    def test_invalid_asset(self, mock_exists) -> None:
        """Test download_assets with invalid asset name."""
        invalid_filename = "invalid.mp4"

        with pytest.raises(ValueError, match="Invalid asset") as exc_info:
            download_assets(invalid_filename)

        assert "Invalid asset" in str(exc_info.value)
        assert "vehicles.mp4" in str(exc_info.value)

    @patch("pathlib.Path.exists", return_value=True)
    def test_invalid_asset_when_file_exists(self, mock_exists) -> None:
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
        self, mock_exists, mock_md5, mock_download, mock_logger
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
        self, mock_exists, mock_md5, mock_download, mock_logger
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
