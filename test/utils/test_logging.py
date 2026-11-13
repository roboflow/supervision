import logging
import time
from datetime import timedelta

import pytest

from supervision.utils.logging import TimeBetweenOccurrencesFilter, setup_logger


def test_logging_without_setup_logger(capsys: pytest.CaptureFixture[str]):
    logger = logging.getLogger("supervision.test_logging_without_setup_logger")
    logger.warning("Info message")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_setup_logger(capsys: pytest.CaptureFixture[str]):
    setup_logger()
    logger = logging.getLogger("supervision.test_setup_logger")
    logger.warning("Info message")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Info message" in captured.err


def function_that_logs(logger: logging.Logger, i: int):
    logger.warning("Info message %d", i)


def test_time_between_occurrences_filter(capsys: pytest.CaptureFixture[str]):
    setup_logger()
    logger = logging.getLogger("supervision.test_time_between_occurrences_filter")
    logger.addFilter(TimeBetweenOccurrencesFilter(timedelta(milliseconds=100)))

    function_that_logs(logger, 0)
    captured = capsys.readouterr()
    assert "Info message 0" in captured.err

    function_that_logs(logger, 1)
    captured = capsys.readouterr()
    assert "Info message 1" not in captured.err

    time.sleep(0.1)
    function_that_logs(logger, 2)
    captured = capsys.readouterr()
    assert "Info message 2" in captured.err
