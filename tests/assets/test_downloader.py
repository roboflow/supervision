from unittest.mock import MagicMock, mock_open, patch

import pytest

from supervision.assets.downloader import download_assets, is_md5_hash_matching
from supervision.assets.list import ImageAssets, VideoAssets


class TestMD5HashMatching:
    def test_file_exists_matching_hash(self):
        """Test is_md5_hash_matching when file exists and hash matches."""
        test_content = b"test content"
        test_hash = "9473fdd0d880a43c21b7778d34872157"  # MD5 of "test content"

        with (
            patch("builtins.open", mock_open(read_data=test_content)),
            patch("os.path.exists", return_value=True),
        ):
            assert is_md5_hash_matching("dummy_file", test_hash)

    def test_file_exists_not_matching_hash(self):
        """Test is_md5_hash_matching when file exists but hash doesn't match."""
        test_content = b"test content"
        wrong_hash = "wrong_hash"

        with (
            patch("builtins.open", mock_open(read_data=test_content)),
            patch("os.path.exists", return_value=True),
        ):
            assert not is_md5_hash_matching("dummy_file", wrong_hash)

    def test_file_not_exists(self):
        """Test is_md5_hash_matching when file doesn't exist."""
        with patch("os.path.exists", return_value=False):
            assert not is_md5_hash_matching("nonexistent_file", "some_hash")


class TestDownloadAssets:
    @patch("supervision.assets.downloader.logger")
    @patch("supervision.assets.downloader.is_md5_hash_matching", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_already_exists_and_valid(self, mock_exists, mock_md5, mock_logger):
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
    ):
        """Test download_assets when file exists but is corrupted (re-downloads)."""
        filename = "vehicles.mp4"
        result = download_assets(filename)
        assert result == filename
        mock_logger.warning.assert_called_once_with("File corrupted. Re-downloading...")
        mock_remove.assert_called_once_with(filename)

    @patch("supervision.assets.downloader.logger")
    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("supervision.assets.downloader.copyfileobj")
    @patch("supervision.assets.downloader.tqdm")
    @patch("supervision.assets.downloader.get")
    def test_download_new_file(
        self,
        mock_get,
        mock_tqdm,
        mock_copyfileobj,
        mock_exists,
        mock_mkdir,
        mock_open_file,
        mock_logger,
    ):
        """Test download_assets downloading a new file."""
        filename = "vehicles.mp4"

        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.raw = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_tqdm.wrapattr.return_value.__enter__ = MagicMock(
            return_value=mock_response.raw
        )
        mock_tqdm.wrapattr.return_value.__exit__ = MagicMock()

        result = download_assets(filename)
        assert result == filename
        mock_logger.info.assert_called_with("Downloading %s assets", filename)
        mock_get.assert_called_once()
        mock_response.raise_for_status.assert_called_once_with()
        mock_copyfileobj.assert_called_once()

    @patch("pathlib.Path.exists", return_value=False)
    def test_invalid_asset(self, mock_exists):
        """Test download_assets with invalid asset name."""
        invalid_filename = "invalid.mp4"

        with pytest.raises(ValueError, match="Invalid asset") as exc_info:
            download_assets(invalid_filename)

        assert "Invalid asset" in str(exc_info.value)
        assert "vehicles.mp4" in str(exc_info.value)

    @patch("pathlib.Path.exists", return_value=True)
    def test_invalid_asset_when_file_exists(self, mock_exists):
        """Test download_assets with invalid asset name that already exists."""
        invalid_filename = "invalid.mp4"

        with pytest.raises(ValueError, match="Invalid asset") as exc_info:
            download_assets(invalid_filename)

        assert "Invalid asset" in str(exc_info.value)
        assert "vehicles.mp4" in str(exc_info.value)

    @patch("supervision.assets.downloader.logger")
    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    @patch("supervision.assets.downloader.copyfileobj")
    @patch("supervision.assets.downloader.tqdm")
    @patch("supervision.assets.downloader.get")
    @patch("pathlib.Path.exists", return_value=False)
    def test_with_video_enum(
        self,
        mock_exists,
        mock_get,
        mock_tqdm,
        mock_copyfileobj,
        mock_mkdir,
        mock_open_file,
        mock_logger,
    ):
        """Test download_assets with VideoAssets enum."""
        asset = VideoAssets.VEHICLES

        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.raw = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_tqdm.wrapattr.return_value.__enter__ = MagicMock()
        mock_tqdm.wrapattr.return_value.__exit__ = MagicMock()

        result = download_assets(asset)
        assert result == asset.filename
        mock_logger.info.assert_called_with("Downloading %s assets", asset.filename)

    @patch("supervision.assets.downloader.logger")
    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    @patch("supervision.assets.downloader.copyfileobj")
    @patch("supervision.assets.downloader.tqdm")
    @patch("supervision.assets.downloader.get")
    @patch("pathlib.Path.exists", return_value=False)
    def test_with_image_enum(
        self,
        mock_exists,
        mock_get,
        mock_tqdm,
        mock_copyfileobj,
        mock_mkdir,
        mock_open_file,
        mock_logger,
    ):
        """Test download_assets with ImageAssets enum."""
        asset = ImageAssets.SOCCER

        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "100"}
        mock_response.raw = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_tqdm.wrapattr.return_value.__enter__ = MagicMock()
        mock_tqdm.wrapattr.return_value.__exit__ = MagicMock()

        result = download_assets(asset)
        assert result == asset.filename
        mock_logger.info.assert_called_with("Downloading %s assets", asset.filename)
