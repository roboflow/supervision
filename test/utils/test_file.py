import os
from contextlib import ExitStack as DoesNotRaise
from typing import Optional

import pytest

from supervision.utils.file import read_txt_file

FILE_1_CONTENT = """Line 1
Line 2
Line 3
"""

FILE_2_CONTENT = """   
Line 2

Line 4

"""  # noqa

FILE_3_CONTENT = """
Line 2

Line 4

"""


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_files():
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
    "file_name, skip_empty, expected_result, exception",
    [
        ("file_1.txt", False, ["Line 1", "Line 2", "Line 3"], DoesNotRaise()),
        ("file_2.txt", True, ["Line 2", "Line 4"], DoesNotRaise()),
        ("file_2.txt", False, ["   ", "Line 2", "", "Line 4", ""], DoesNotRaise()),
        ("file_3.txt", True, ["Line 2", "Line 4"], DoesNotRaise()),
        ("file_3.txt", False, ["", "Line 2", "", "Line 4", ""], DoesNotRaise()),
        ("file_4.txt", True, None, pytest.raises(FileNotFoundError)),
    ],
)
def test_read_txt_file(
    file_name: str,
    skip_empty: bool,
    expected_result: Optional[list[str]],
    exception: Exception,
):
    with exception:
        result = read_txt_file(file_name, skip_empty)
        assert result == expected_result
