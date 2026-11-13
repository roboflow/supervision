import json
import os
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from contextlib import ExitStack as DoesNotRaise
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests

from supervision.utils.file import (
    _download_to_file,
    _normalize_http_url,
    list_files_with_extensions,
    read_json_file,
    read_txt_file,
    read_yaml_file,
    save_json_file,
)


class TestNormalizeHttpUrl:
    def test_returns_normalized_http_url(self) -> None:
        """Valid HTTPS URL is returned normalized."""
        # given
        url = "https://media.roboflow.com/quickstart/dog.jpeg"

        # when
        result = _normalize_http_url(url=url)

        # then
        assert result == url

    @pytest.mark.parametrize(
        ("url", "match"),
        [
            pytest.param(
                "file:///tmp/image.jpg", "Unsupported URL scheme", id="file-scheme"
            ),
            pytest.param(
                "ftp://example.com/image.jpg",
                "Unsupported URL scheme",
                id="ftp-scheme",
            ),
            pytest.param(
                "javascript:alert(1)",
                "Unsupported URL scheme",
                id="javascript-scheme",
            ),
            pytest.param(
                "data:text/plain;base64,aGk=",
                "Unsupported URL scheme",
                id="data-scheme",
            ),
            pytest.param("not a url", "Invalid URL", id="not-a-url"),
            pytest.param("http://", "Invalid URL", id="missing-host"),
            pytest.param(
                "https://foo\\bar/image.jpg", "Invalid URL", id="backslash-authority"
            ),
        ],
    )
    def test_rejects_invalid_url(self, url: str, match: str) -> None:
        """Invalid or non-HTTP(S) URLs raise ValueError."""
        with pytest.raises(ValueError, match=match):
            _normalize_http_url(url=url)


class TestDownloadToFile:
    def test_writes_response_content_to_target(self, tmp_path: Path) -> None:
        """Non-streaming download writes response bytes to the target path."""
        # given
        target = tmp_path / "subdir" / "file.bin"
        response = Mock()
        response.content = b"payload"
        response.raise_for_status.return_value = None

        # when
        with patch("supervision.utils.file.requests.get", return_value=response) as get:
            _download_to_file("https://example.com/file.bin", target)

        # then
        get.assert_called_once_with(
            "https://example.com/file.bin",
            stream=False,
            allow_redirects=True,
            timeout=30.0,
        )
        assert target.read_bytes() == b"payload"
        assert list(target.parent.iterdir()) == [target]
        response.close.assert_called_once()

    def test_raises_and_leaves_no_file_on_http_error(self, tmp_path: Path) -> None:
        """HTTP error status raises and leaves no file behind."""
        # given
        target = tmp_path / "file.bin"
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("404")

        # when / then
        with (
            patch("supervision.utils.file.requests.get", return_value=response),
            pytest.raises(requests.HTTPError, match="404"),
        ):
            _download_to_file("https://example.com/file.bin", target)

        assert not target.exists()
        response.close.assert_called_once()


FILE_1_CONTENT = """Line 1
Line 2
Line 3
"""

FILE_2_CONTENT = """   \nLine 2

Line 4

"""

FILE_3_CONTENT = """
Line 2

Line 4

"""


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_files() -> Iterator[None]:
    """Create the text fixtures used by the read tests and remove them after."""
    with open("file_1.txt", "w") as file:
        file.write(FILE_1_CONTENT)
    with open("file_2.txt", "w") as file:
        file.write(FILE_2_CONTENT)
    with open("file_3.txt", "w") as file:
        file.write(FILE_3_CONTENT)

    yield

    os.remove("file_1.txt")
    os.remove("file_2.txt")
    os.remove("file_3.txt")


@pytest.mark.parametrize(
    ("file_name", "skip_empty", "expected_result", "exception"),
    [
        ("file_1.txt", False, ["Line 1", "Line 2", "Line 3"], DoesNotRaise()),
        ("file_2.txt", True, ["Line 2", "Line 4"], DoesNotRaise()),
        ("file_2.txt", False, ["   ", "Line 2", "", "Line 4", ""], DoesNotRaise()),
        ("file_3.txt", True, ["Line 2", "Line 4"], DoesNotRaise()),
        ("file_3.txt", False, ["", "Line 2", "", "Line 4", ""], DoesNotRaise()),
        (
            "file_4.txt",
            True,
            None,
            pytest.raises(FileNotFoundError, match=r"file_4\.txt"),
        ),
    ],
)
def test_read_txt_file(
    file_name: str,
    skip_empty: bool,
    expected_result: list[str] | None,
    exception: AbstractContextManager[object],
) -> None:
    """Text files are read line by line, optionally skipping empty lines."""
    with exception:
        result = read_txt_file(file_name, skip_empty)
        assert result == expected_result


@pytest.mark.parametrize(
    ("filenames_to_create", "extension", "expected_names"),
    [
        pytest.param(
            ["image.jpg", "image.png"], ".jpg", {"image.jpg"}, id="leading-dot"
        ),
        pytest.param(["image.JPG"], "jpg", {"image.JPG"}, id="case-insensitive"),
        pytest.param(
            ["archive.tar.gz"], "tar.gz", {"archive.tar.gz"}, id="multi-part-full"
        ),
        pytest.param(
            ["archive.backup.tar.gz", "archive.backup.gz"],
            "tar.gz",
            {"archive.backup.tar.gz"},
            id="multi-part-filename-tail",
        ),
        pytest.param(
            ["archive.tar.gz", "data.gz"],
            "gz",
            {"archive.tar.gz", "data.gz"},
            id="multi-part-suffix",
        ),
    ],
)
def test_list_files_with_extensions_normalization(
    tmp_path: Path,
    filenames_to_create: list[str],
    extension: str,
    expected_names: set[str],
) -> None:
    """Extension matching normalizes leading dots, case, and multi-part extensions."""
    for filename in filenames_to_create:
        (tmp_path / filename).touch()

    result = list_files_with_extensions(directory=tmp_path, extensions=[extension])

    assert {p.name for p in result} == expected_names


def test_list_files_with_extensions_without_filter_ignores_directories(
    tmp_path: Path,
) -> None:
    """Directory entries are ignored when no extension filter is provided."""
    # given
    (tmp_path / "image.jpg").touch()
    (tmp_path / "labels").mkdir()

    # when
    result = list_files_with_extensions(directory=tmp_path)

    # then
    assert {p.name for p in result} == {"image.jpg"}


@pytest.mark.parametrize(
    ("fn_read_file", "file_name", "content"),
    [
        pytest.param(
            read_json_file,
            "data.json",
            '{"names": ["café", "고양이", "猫"]}',
            id="json",
        ),
        pytest.param(
            read_yaml_file,
            "data.yaml",
            "names:\n  - café\n  - 고양이\n  - 猫\n",
            id="yaml",
        ),
    ],
)
def test_read_structured_file_decodes_utf8_text(
    tmp_path: Path,
    fn_read_file: Callable[[Path], dict[str, Any]],
    file_name: str,
    content: str,
) -> None:
    """Non-ASCII text in a UTF-8 file reads back unchanged on every platform."""
    file_path = tmp_path / file_name
    file_path.write_bytes(content.encode("utf-8"))

    data = fn_read_file(file_path)

    assert data == {"names": ["café", "고양이", "猫"]}


class TestSaveJsonFile:
    @pytest.mark.parametrize("value", [True, False])
    def test_saves_numpy_booleans(self, tmp_path: Path, value: bool) -> None:
        """NumPy booleans remain JSON booleans, including in nested fields."""
        file_path = tmp_path / "data.json"
        data = {"flag": np.bool_(value), "nested": {"flag": np.bool_(value)}}

        save_json_file(data, file_path)

        result = json.loads(file_path.read_text(encoding="utf-8"))
        assert result["flag"] is value
        assert result["nested"]["flag"] is value

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, True),
            (False, False),
            pytest.param(np.int64(7), 7, id="numpy-integer"),
            pytest.param(np.float32(0.5), 0.5, id="numpy-float"),
            pytest.param(np.array([1, 2]), [1, 2], id="numpy-array"),
            pytest.param(
                np.array([True, False]), [True, False], id="numpy-boolean-array"
            ),
        ],
    )
    def test_preserves_supported_values(
        self, tmp_path: Path, value: Any, expected: Any
    ) -> None:
        """Existing scalar and array values retain their JSON types and values."""
        file_path = tmp_path / "data.json"

        save_json_file({"value": value}, file_path)

        result = json.loads(file_path.read_text(encoding="utf-8"))["value"]
        assert result == expected
        assert type(result) is type(expected)
