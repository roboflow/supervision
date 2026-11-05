import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple


def setup_logger(**levels: int):
    formatter = logging.Formatter(
        "%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    root = logging.getLogger("supervision")
    root.setLevel(logging.WARNING)
    root.addHandler(console_handler)

    for logger_name, logger_level in levels.items():
        logging.getLogger(logger_name).setLevel(logger_level)


_LogId = Tuple[str, int]


def _get_record_id(record: logging.LogRecord) -> _LogId:
    return record.pathname, record.lineno


class TimeBetweenOccurrencesFilter(logging.Filter):
    """
    Adds filtering based on time elapsed between two same logging calls.
    Useful to prevent displaying too much messages, e.g. in the loop.
    """

    def __init__(self, min_interval: timedelta):
        super().__init__()
        self._min_interval = min_interval

        self._last_activity: Dict[_LogId, datetime] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        record_id = _get_record_id(record)
        now = datetime.fromtimestamp(record.created)
        last_activity = self._last_activity.get(record_id)
        self._last_activity[record_id] = now
        if last_activity is not None:
            return now - last_activity >= self._min_interval
        return True
