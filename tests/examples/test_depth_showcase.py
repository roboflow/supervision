"""Regression tests for the depth showcase example."""

from __future__ import annotations

import itertools
import sys
import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from examples.depth_showcase import depth_showcase


def test_file_source_reads_to_eof_when_frame_count_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep file reads unbounded when metadata cannot provide a safe end frame."""
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    calls = []

    def read_frames(path: str, end: int | None = None) -> Iterator[np.ndarray]:
        calls.append((path, end))
        yield frame

    monkeypatch.setattr(depth_showcase.sv, "get_video_frames_generator", read_frames)
    source = depth_showcase.VideoSource("clip.mp4", 30, None, None)

    assert list(source.frames()) == [frame]
    assert calls == [("clip.mp4", None)]


def test_main_reads_real_video_to_eof_when_metadata_undercounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decode every real frame when file metadata reports a smaller count."""
    video_path = tmp_path / "underreported.mp4"
    with av.open(str(video_path), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=5)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for value in (20, 60, 100, 140, 180):
            image = np.full((16, 16, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    observed = {}
    monkeypatch.setattr(
        depth_showcase.sv.VideoInfo,
        "from_video_path",
        lambda _path: depth_showcase.sv.VideoInfo(16, 16, 5.0, 1),
    )
    monkeypatch.setattr(
        depth_showcase,
        "run_model",
        lambda _name, source, *_args: observed.setdefault("source", source) and True,
    )

    depth_showcase.main(str(video_path), models=["probe"])
    frames = list(observed["source"].frames())

    assert len(frames) == 5


@pytest.mark.parametrize(
    ("max_frames", "error"),
    [
        pytest.param(0, "greater than zero", id="zero-frame-limit"),
        pytest.param(-1, "greater than zero", id="negative-frame-limit"),
    ],
)
def test_video_source_rejects_nonpositive_frame_limits(
    max_frames: int, error: str
) -> None:
    """Reject frame limits that would silently produce no useful work."""
    with pytest.raises(ValueError, match=error):
        depth_showcase.VideoSource("clip.mp4", 30, None, max_frames)


def test_main_rejects_duration_shorter_than_one_frame() -> None:
    """Explain when an explicit duration rounds down to zero frames."""
    with pytest.raises(ValueError, match="at least one frame"):
        depth_showcase.main("0", max_seconds=0.001, models=[])


def test_empty_source_raises_a_clear_error_before_loading_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an empty source without leaking StopIteration or loading a model."""
    source = depth_showcase.VideoSource("0", 30, None, 1)
    monkeypatch.setattr(source, "frames", lambda: iter(()))
    monkeypatch.setattr(
        depth_showcase,
        "MODELS",
        {"model": lambda _device: pytest.fail("empty sources must fail first")},
    )

    with pytest.raises(ValueError, match="no readable frames"):
        depth_showcase.run_model("model", source, None, "turbo", 0.9, 5, "cpu")


def test_invalid_depth_map_keeps_the_previous_running_range() -> None:
    """Avoid asking clip-range estimation to process a frame with no valid depth."""
    invalid = depth_showcase.sv.DepthMap(
        np.full((2, 2), np.nan, dtype=np.float32), kind="relative_inverse"
    )

    assert depth_showcase.blend_range((1.0, 2.0), invalid) == (1.0, 2.0)
    assert depth_showcase.blend_range(None, invalid) is None


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        pytest.param(True, True, "cuda", id="cuda-preferred"),
        pytest.param(False, True, "mps", id="mps-fallback"),
        pytest.param(False, False, "cpu", id="cpu-fallback"),
    ],
)
def test_default_device_uses_available_torch_backends(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    """Select an accelerator only when the installed backend reports availability."""
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert depth_showcase.default_device() == expected


def test_inference_caption_uses_running_average_after_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclude the first prediction from a constant-memory running speed average."""
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
    source = depth_showcase.VideoSource("0", 30, None, len(frames))
    monkeypatch.setattr(source, "frames", lambda: iter(frames))
    depth_map = depth_showcase.sv.DepthMap(
        np.ones((2, 2), dtype=np.float32), kind="relative_inverse"
    )
    monkeypatch.setitem(
        depth_showcase.MODELS,
        "model",
        lambda _device: (lambda _frame: depth_map, "model"),
    )
    monkeypatch.setattr(depth_showcase, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(
        depth_showcase.sv,
        "DepthAnnotator",
        lambda **_kwargs: SimpleNamespace(annotate=lambda scene, _depth: scene),
    )
    window = SimpleNamespace(
        close=lambda: None,
        show=lambda _frame: None,
        is_open=True,
        wait_key=lambda _delay: "",
    )
    monkeypatch.setattr(depth_showcase.sv, "ImageWindow", lambda _name: window)
    monkeypatch.setattr(depth_showcase.sv, "Point", lambda *_args: None)
    captions = []
    monkeypatch.setattr(
        depth_showcase.sv,
        "draw_text",
        lambda frame, text, *_args, **_kwargs: captions.append(text) or frame,
    )
    elapsed = iter((0.0, 0.1, 1.0, 1.2, 2.0, 2.4))
    monkeypatch.setattr(depth_showcase.time, "perf_counter", lambda: next(elapsed))

    assert depth_showcase.run_model("model", source, None, "turbo", 0.9, 5, "cpu")
    assert [caption.rsplit(" | ", 1)[1] for caption in captions] == [
        "100 ms/frame",
        "200 ms/frame",
        "300 ms/frame",
    ]


def test_inference_timing_memory_is_bounded_by_frame_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep timing state constant-sized while processing a long stream."""
    frame_count = 10_000
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    source = depth_showcase.VideoSource("0", 30, None, frame_count)
    monkeypatch.setattr(source, "frames", lambda: itertools.repeat(frame, frame_count))
    depth_map = depth_showcase.sv.DepthMap(
        np.ones((2, 2), dtype=np.float32), kind="relative_inverse"
    )
    monkeypatch.setitem(
        depth_showcase.MODELS,
        "model",
        lambda _device: (lambda _frame: depth_map, "model"),
    )
    monkeypatch.setattr(depth_showcase, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(
        depth_showcase, "blend_range", lambda _previous, _depth: (1.0, 1.0)
    )
    monkeypatch.setattr(
        depth_showcase.sv,
        "DepthAnnotator",
        lambda **_kwargs: SimpleNamespace(annotate=lambda scene, _depth: scene),
    )
    window = SimpleNamespace(
        close=lambda: None,
        show=lambda _frame: None,
        is_open=True,
        wait_key=lambda _delay: "",
    )
    monkeypatch.setattr(depth_showcase.sv, "ImageWindow", lambda _name: window)
    monkeypatch.setattr(depth_showcase.sv, "Point", lambda *_args: None)
    monkeypatch.setattr(
        depth_showcase.sv, "draw_text", lambda frame, *_args, **_kwargs: frame
    )
    clock = itertools.count()
    monkeypatch.setattr(depth_showcase.time, "perf_counter", lambda: next(clock) / 1000)

    tracemalloc.start()
    try:
        assert depth_showcase.run_model("model", source, None, "turbo", 0.9, 5, "cpu")
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak_bytes < 150_000
