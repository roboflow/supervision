from __future__ import annotations

import importlib
from collections.abc import Generator
from dataclasses import replace

import numpy as np
from tqdm.auto import tqdm

from .utils import VideoInfo


class Video:
    def __init__(self, video_path: str, backend: str | None = None):
        self.video_path = video_path
        self._backend_name = backend or "opencv"
        self._backend = self.__get_backend()

    def __len__(self) -> int:
        return len(self._backend)

    def __iter__(self):
        return iter(self._backend)

    def __getitem__(self, index: int) -> np.ndarray:
        return self._backend[index]

    def __repr__(self) -> str:
        return f"<Video {self.video_path} : {self.info}>"

    def __get_backend(self):
        from .backends import BACKENDS

        try:
            module_path = BACKENDS[self._backend_name]
        except KeyError:
            raise ValueError(
                f"Unknown backend '{self._backend_name}'. "
                f"Available backends: {', '.join(BACKENDS.keys())}"
            )
        module = importlib.import_module(module_path)
        self._backend = module.Backend(str(self.video_path))
        return self._backend

    @property
    def info(self) -> VideoInfo:
        return self._backend.info()

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
    ) -> Generator[np.ndarray]:
        yield from self._backend.frames(stride, start, end, resolution_wh)

    def save(
        self,
        path: str,
        callback=None,
        show_progress=True,
        info: VideoInfo | None = None,
        **kwargs,
    ):
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
