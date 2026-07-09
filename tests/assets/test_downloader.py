from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from supervision.assets.downloader import download_assets, is_md5_hash_matching
from supervision.assets.list import MEDIA_ASSETS, ImageAssets, VideoAssets


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
    @patch(
        "supervision.assets.downloader.is_md5_hash_matching",
        side_effect=[False, True],
    )
    @patch("pathlib.Path.exists", return_value=True)
    def test_already_exists_but_corrupted(
        self, mock_exists, mock_md5, mock_remove, mock_logger
    ) -> None:
        """Test download_assets when file exists but is corrupted (re-downloads)."""
        filename = "vehicles.mp4"
        result = download_assets(filename)
        assert result == filename
        mock_logger.warning.assert_called_once_with("File corrupted. Re-downloading...")
        mock_remove.assert_called_once_with(filename)

    @patch("supervision.assets.downloader.logger")
    @patch("supervision.assets.downloader._download_to_file")
    @patch("pathlib.Path.exists", return_value=False)
    def test_download_new_file(self, mock_exists, mock_download, mock_logger) -> None:
        """Test download_assets downloading a new file."""
        filename = "vehicles.mp4"

        result = download_assets(filename)

        assert result == filename
        mock_logger.info.assert_called_with("Downloading %s assets", filename)
        mock_download.assert_called_once_with(
            MEDIA_ASSETS[filename][0],
            Path(filename).expanduser().resolve(),
            timeout=30.0,
            stream=True,
        )

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
    @patch("supervision.assets.downloader._download_to_file")
    @patch("pathlib.Path.exists", return_value=False)
    def test_with_video_enum(self, mock_exists, mock_download, mock_logger) -> None:
        """Test download_assets with VideoAssets enum."""
        asset = VideoAssets.VEHICLES

        result = download_assets(asset)

        assert result == asset.filename
        mock_logger.info.assert_called_with("Downloading %s assets", asset.filename)
        mock_download.assert_called_once()

    @patch("supervision.assets.downloader.logger")
    @patch("supervision.assets.downloader._download_to_file")
    @patch("pathlib.Path.exists", return_value=False)
    def test_with_image_enum(self, mock_exists, mock_download, mock_logger) -> None:
        """Test download_assets with ImageAssets enum."""
        asset = ImageAssets.SOCCER

        result = download_assets(asset)

        assert result == asset.filename
        mock_logger.info.assert_called_with("Downloading %s assets", asset.filename)
        mock_download.assert_called_once()
