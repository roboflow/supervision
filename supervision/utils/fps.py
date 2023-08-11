import time
from collections import deque


class FpsMonitor:
    def __init__(self, sample_size: int = 30):
        """
        Initialize class to measure Frame per second for latency benchmark.
        Args:
            sample_size (int): Maximum observation to measure latency benchmark
        Examples:
            ```python
            >>> import supervision as sv

            >>> fps_monitor = sv.FpsMonitor()
            >>> while True:
            ...      (...)
            ...      fps_monitor.tick()
            ...      fps = fps_monitor()
            ```
        """
        self.all_timestamps = deque(maxlen=sample_size)

    def __call__(self) -> float:
        if not self.all_timestamps:
            return 0.0
        taken_time = self.all_timestamps[-1] - self.all_timestamps[0]
        return (len(self.all_timestamps)) / taken_time if taken_time != 0 else 0.0

    def tick(self) -> None:
        self.all_timestamps.append(time.monotonic())

    def reset(self) -> None:
        self.all_timestamps.clear()