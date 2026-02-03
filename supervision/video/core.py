from __future__ import annotations

import importlib
from collections.abc import Generator
from dataclasses import replace

import numpy as np
from tqdm.auto import tqdm

from .utils import VideoInfo


class Video:
    def __init__(self, video_path: str, backend: str | None = None):
        """High-level video reader and writer.

        This class provides a unified interface over pluggable backends for
        reading frames, iterating, slicing, and saving videos.

        Args:
            video_path: Path or identifier of the source video. This can be a
                file path, URL, or camera index depending on the backend.
            backend: Optional backend name (for example, ``"pyav"``). If not
                provided, ``"opencv"`` is used.
        """
        self.video_path = video_path
        self._backend_name = backend or "opencv"
        self._backend = self.__get_backend()

    def __len__(self) -> int:
        """Return the number of frames if known.

        Returns:
            int: The total number of frames.

        Raises:
            TypeError: If the underlying stream does not expose a finite length.
        """
        return len(self._backend)

    def __iter__(self):
        """Return an iterator over decoded frames as ``np.ndarray``.

        Yields:
            np.ndarray: The next decoded frame in BGR/RGB format depending on
                the backend.
        """
        return iter(self._backend)

    def __getitem__(self, index: int) -> np.ndarray:
        """Return the frame at a specific index.

        Args:
            index: Zero-based frame index to retrieve.

        Returns:
            np.ndarray: The decoded frame at ``index``.

        Raises:
            IndexError: If the index is out of bounds or cannot be read.
        """
        return self._backend[index]

    def __repr__(self) -> str:
        """Return a concise representation including path and info."""
        return (
            f"<Video {self.video_path} : {self.info} : Backend: {self._backend_name}>"
        )

    def __get_backend(self):
        """Instantiate the selected backend implementation.

        Returns:
            Backend: The instantiated backend object matching ``self._backend_name``.

        Raises:
            ValueError: If an unknown backend name is provided.
        """
        from .backends import BACKENDS

        try:
            module_path = BACKENDS[self._backend_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown backend '{self._backend_name}'. "
                f"Available backends: {', '.join(BACKENDS.keys())}"
            ) from exc
        module = importlib.import_module(module_path)
        self._backend = module.Backend(str(self.video_path))
        return self._backend

    @property
    def info(self) -> VideoInfo:
        """Return static information about the video source.

        Returns:
            VideoInfo: Width, height, frames per second, and total frames when
            available.
        """
        return self._backend.info()

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
    ) -> Generator[np.ndarray]:
        """Yield frames lazily with optional skipping and resizing.

        Args:
            stride: Number of frames to skip between yielded frames (``1`` yields
                every frame).
            start: First frame index (0-based) to yield.
            end: Index after the last frame to yield. ``None`` means until
                exhaustion.
            resolution_wh: Optional ``(width, height)`` to resize each yielded
                frame to.

        Yields:
            np.ndarray: The next decoded (and optionally resized) video frame.
        """
        yield from self._backend.frames(stride, start, end, resolution_wh)

    def save(
        self,
        path: str,
        callback=None,
        show_progress=True,
        info: VideoInfo | None = None,
        **kwargs,
    ):
        """Encode and save frames from this video to a new file.

        Args:
            path: Output file path.
            callback: Optional callable to transform frames before writing. It
                receives ``(frame, frame_number)`` and should return a frame.
            show_progress: If ``True``, display a progress bar while saving.
            info: Optional ``VideoInfo`` describing the desired output
                properties. If omitted, the source video info is used.
            **kwargs: Additional ``VideoInfo`` fields to override (for example,
                ``fps``, ``width``, ``height``). Only keys present in
                ``VideoInfo`` are applied.

        Returns:
            None
        """
        updated_info = info or self.info
        updated_info = replace(
            updated_info,
            **{k: v for k, v in kwargs.items() if k in self.info.__dataclass_fields__},
        )
        with self._backend.writer(path, updated_info) as writer:
            for i, frame in enumerate(
                tqdm(self.frames(), desc="Saving video", disable=not show_progress)
            ):
                writer.write(frame, frame_number=i, callback=callback)
