from __future__ import annotations

import logging
import sys
from unittest.mock import patch

from supervision.utils.logger import _get_logger


class TestGetLogger:
    def test_default_name(self):
        """Logger is created with default name."""
        logger = _get_logger()
        assert logger.name == "supervision"

    def test_custom_name(self):
        """Logger is created with provided name."""
        logger = _get_logger("supervision.test_module")
        assert logger.name == "supervision.test_module"

    def test_default_level_is_info(self):
        """Logger defaults to INFO level when LOG_LEVEL env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Use a unique name to avoid cached logger state from other tests
            logger = _get_logger("supervision.test_default_level")
        assert logger.level == logging.INFO

    def test_explicit_level(self):
        """Logger uses the explicitly provided level."""
        logger = _get_logger("supervision.test_explicit_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_log_level_env_var(self):
        """Logger respects the LOG_LEVEL environment variable."""
        with patch.dict("os.environ", {"LOG_LEVEL": "DEBUG"}):
            logger = _get_logger("supervision.test_env_level")
        assert logger.level == logging.DEBUG

    def test_log_level_env_var_warning(self):
        """Logger respects the LOG_LEVEL=WARNING environment variable."""
        with patch.dict("os.environ", {"LOG_LEVEL": "WARNING"}):
            logger = _get_logger("supervision.test_env_warning")
        assert logger.level == logging.WARNING

    def test_two_handlers_configured(self):
        """Logger has exactly two handlers: one for stdout, one for stderr."""
        logger = _get_logger("supervision.test_handlers")
        assert len(logger.handlers) == 2

    def test_stdout_handler_present(self):
        """Logger has a StreamHandler pointing to stdout."""
        logger = _get_logger("supervision.test_stdout")
        stdout_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
        ]
        assert len(stdout_handlers) == 1

    def test_stderr_handler_present(self):
        """Logger has a StreamHandler pointing to stderr for warnings."""
        logger = _get_logger("supervision.test_stderr")
        stderr_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
        ]
        assert len(stderr_handlers) == 1

    def test_no_propagation(self):
        """Logger does not propagate to the root logger."""
        logger = _get_logger("supervision.test_propagation")
        assert not logger.propagate

    def test_idempotent_no_duplicate_handlers(self):
        """Calling _get_logger twice with the same name does not add extra handlers."""
        name = "supervision.test_idempotent"
        logger1 = _get_logger(name)
        handler_count = len(logger1.handlers)
        logger2 = _get_logger(name)
        assert len(logger2.handlers) == handler_count
        assert logger1 is logger2
