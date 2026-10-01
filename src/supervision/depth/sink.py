from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

from supervision.depth.core import (
    DepthClipRange,
    DepthKind,
    DepthMap,
    _manifest_header,
)
from supervision.depth.manifest import (
    PREVIEW_TOP_CODES,
    avc_codec_string,
    encode_codes,
    encode_png16,
    power_of_two_scale,
)
from supervision.utils.logger import _get_logger
from supervision.utils.video import VideoInfo

logger = _get_logger(__name__)

_EXACT_PATTERN = "exact/{index:06}.png"
_PREVIEW_FILE = "preview.mp4"
_TV_BLACK = 16
_TV_TOP = PREVIEW_TOP_CODES["tv"]
_NEUTRAL_CHROMA = 128
# libavutil enum values: AVCOL_RANGE_MPEG (TV range) and BT.709 for the primaries,
# the transfer characteristic and the matrix.
_AVCOL_RANGE_MPEG = 1
_AVCOL_BT709 = 1


def _is_int(value: Any) -> bool:
    """Report a Python `int` that is not a `bool`."""
    return isinstance(value, int) and not isinstance(value, bool)


def _frame_rate(fps: float) -> Fraction:
    """Return the frame rate as an exact fraction, snapping NTSC rates to x/1001.

    `sv.VideoInfo` reports 30000/1001 as 29.97 or 29.97002997...; written as 2997/100
    the preview's frame times would drift from the video's by about 0.04 ms a
    minute, and supervision-js rejects a preview frame more than 0.5 ms off its
    video frame.
    """
    if not fps > 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    ntsc = round(fps * 1.001)
    if abs(fps - ntsc / 1.001) < 1e-3 and abs(fps - round(fps)) > 1e-3:
        return Fraction(ntsc * 1000, 1001)
    return Fraction(str(fps)).limit_denominator(100_000)


def _preview_codes(
    depth_map: DepthMap,
    value_range: tuple[float, float],
    reserved_max: int,
) -> npt.NDArray[np.uint8]:
    """Return a map's 8-bit TV-range preview codes, as supervision-js decodes them.

    No depth is code 16 (TV black). Codes up to `reserved_max` are a guard band that
    keeps the codec's error around holes reading as no depth, and depth `d` is
    `clamp(T + 1 + rint((d - low) / (high - low) * (235 - T - 1)), T + 1, 235)` with
    `T = reserved_max`.
    """
    low, high = value_range
    span = _TV_TOP - reserved_max - 1
    values = depth_map.to_float(no_depth_value=low).astype(np.float64)
    codes: npt.NDArray[np.uint8] = np.clip(
        reserved_max + 1 + np.rint((values - low) / (high - low) * span),
        reserved_max + 1,
        _TV_TOP,
    ).astype(np.uint8)
    codes[~depth_map.valid_mask] = _TV_BLACK
    return codes


class DepthSink:
    """Write a clip of depth maps in the format supervision-js plays.

    The sink writes, in `target_dir`:

    - `exact/000000.png`, `exact/000001.png`, ...: one 16-bit PNG per frame with the
      exact values, quantised with the largest power-of-two scale that holds the
      clip range's `max_value`.
    - `preview.mp4`: an 8-bit H.264 video of the same frames, which supervision-js
      draws while the clip plays. Codes sit in the luma of `yuv420p` with neutral
      chroma, in TV range flagged with BT.709 colour, at CRF 18 with a keyframe every
      second; frame `k` is at `k / fps`, the time of video frame `k`. The preview
      spans `[0, max_value]` and is defined for `disparity_px` maps only.
    - `depth.json`: the clip manifest, written when the `with` block exits without an
      error, with the clip range's `display_range`.

    Frames are matched to the video by index, so write exactly one map per video
    frame, in order. Every map must have the first map's kind and size; the
    manifest's camera and view come from the first map.

    Attributes:
        target_dir: Folder the clip is written to.
        video_info: The video the depth belongs to; its `fps` times the preview.
        clip_range: The clip's colour range and largest value, from
            `sv.DepthClipRange.from_depth_maps` or written by hand from a known
            range, such as a stereo matcher's disparity search range.

    Examples:
        ```python
        import supervision as sv

        video_info = sv.VideoInfo.from_video_path("left.mp4")
        depth_maps = [model_depth(frame) for frame in sv.get_video_frames_generator("left.mp4")]
        clip_range = sv.DepthClipRange.from_depth_maps(depth_maps)

        with sv.DepthSink("left-depth", video_info, clip_range) as sink:
            for depth_map in depth_maps:
                sink.write_depth_map(depth_map)
        ```
    """  # noqa: E501 // docs

    def __init__(
        self,
        target_dir: str | Path,
        video_info: VideoInfo,
        clip_range: DepthClipRange,
        preview: bool = True,
        crf: int = 18,
        reserved_max: int = 31,
    ) -> None:
        """
        Args:
            target_dir: Folder to write the clip to; created if missing.
            video_info: Information about the video the depth belongs to.
            clip_range: The clip's colour range and largest value.
            preview: Whether to write `preview.mp4`. Without it, supervision-js draws
                exact depth only while playback rests.
            crf: H.264 constant rate factor of the preview; lower is closer to the
                exact codes, 0 is lossless.
            reserved_max: Highest preview code meaning no depth, from 16 to 233. Each
                code of guard band costs one step of preview precision.

        Raises:
            ValueError: If `reserved_max` or `crf` is not an integer in its range.
        """
        if not _is_int(reserved_max) or not 16 <= reserved_max <= _TV_TOP - 2:
            raise ValueError(
                f"reserved_max must be an integer from 16 to {_TV_TOP - 2}, got "
                f"{reserved_max!r}."
            )
        if not _is_int(crf) or not 0 <= crf <= 51:
            raise ValueError(f"crf must be an integer from 0 to 51, got {crf!r}.")
        self.target_dir = Path(target_dir)
        self.video_info = video_info
        self.clip_range = clip_range
        self.preview = preview
        self.crf = crf
        self.reserved_max = reserved_max
        self._rate = _frame_rate(video_info.fps)
        self._scale = power_of_two_scale(clip_range.max_value)
        self._first: DepthMap | None = None
        self._count = 0
        self._container: Any = None
        self._stream: Any = None
        self._is_open = False

    def __enter__(self) -> DepthSink:
        """Create the target folders for a new clip and drop an earlier manifest."""
        (self.target_dir / "exact").mkdir(parents=True, exist_ok=True)
        # A manifest left by an earlier clip would describe frames this one overwrites.
        (self.target_dir / "depth.json").unlink(missing_ok=True)
        self._first = None
        self._count = 0
        self._is_open = True
        return self

    def write_depth_map(self, depth_map: DepthMap) -> None:
        """Write the clip's next frame.

        Args:
            depth_map: The depth for the next video frame.

        Raises:
            ValueError: If the map's kind or size differs from the first map's, a
                value does not fit the clip's scale, or the preview cannot be written
                for this map (a kind other than `disparity_px`, or an odd size).
            RuntimeError: If the sink is used outside a `with` block.
        """
        if not self._is_open:
            raise RuntimeError("write_depth_map requires an open DepthSink context.")
        if self._first is None:
            self._open(depth_map)
        first = self._first
        assert first is not None
        if depth_map.kind is not first.kind or (
            depth_map.resolution_wh != first.resolution_wh
        ):
            raise ValueError(
                f"Frame {self._count} is a {depth_map.kind.value} map of "
                f"{depth_map.resolution_wh}, but the clip is {first.kind.value} at "
                f"{first.resolution_wh}."
            )
        if depth_map.values.dtype == np.uint16 and depth_map.scale == self._scale:
            codes = depth_map.values
        else:
            codes = encode_codes(
                depth_map.to_float(0.0), depth_map.valid_mask, self._scale
            )
        exact_path = self.target_dir / _EXACT_PATTERN.format(index=self._count)
        exact_path.write_bytes(encode_png16(codes))
        if self._stream is not None:
            self._encode_preview(depth_map)
        self._count += 1

    def _open(self, depth_map: DepthMap) -> None:
        """Take the clip's kind and size from its first map and open the preview."""
        width, height = depth_map.resolution_wh
        if self.preview and depth_map.kind is not DepthKind.DISPARITY_PX:
            raise ValueError(
                "The preview video is defined for disparity_px maps only; convert "
                "with depth_map.to_disparity() or pass preview=False."
            )
        if self.preview and (width % 2 or height % 2):
            raise ValueError(
                f"The yuv420p preview needs an even width and height, got "
                f"{width}x{height}; resize the maps or pass preview=False."
            )
        self._first = depth_map
        if not self.preview:
            return
        import av

        container = av.open(
            str(self.target_dir / _PREVIEW_FILE),
            mode="w",
            options={"movflags": "+faststart"},
        )
        try:
            stream = container.add_stream("libx264", rate=self._rate)
        except Exception as error:
            container.close()
            raise RuntimeError(
                "PyAV has no libx264 encoder; pass preview=False to write exact "
                "frames only."
            ) from error
        keyframe_interval = max(1, round(float(self._rate)))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        codec_context = stream.codec_context
        codec_context.color_range = _AVCOL_RANGE_MPEG
        codec_context.color_primaries = _AVCOL_BT709
        codec_context.color_trc = _AVCOL_BT709
        codec_context.colorspace = _AVCOL_BT709
        codec_context.gop_size = keyframe_interval
        stream.options = {
            "crf": str(self.crf),
            "tune": "psnr",
            "preset": "medium",
            "x264-params": (
                f"keyint={keyframe_interval}:min-keyint={keyframe_interval}:scenecut=0"
            ),
        }
        self._container = container
        self._stream = stream

    def _encode_preview(self, depth_map: DepthMap) -> None:
        """Encode one preview frame: codes in luma, neutral chroma, pts = index."""
        import av

        width, height = depth_map.resolution_wh
        luma = _preview_codes(
            depth_map, (0.0, self.clip_range.max_value), self.reserved_max
        )
        chroma = np.full((height // 2, width), _NEUTRAL_CHROMA, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(
            np.concatenate([luma, chroma]), format="yuv420p"
        )
        frame.pts = self._count
        frame.time_base = 1 / self._rate
        frame.color_range = _AVCOL_RANGE_MPEG
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        """Flush the preview and write `depth.json`, unless the block raised."""
        container, stream = self._container, self._stream
        self._container = self._stream = None
        self._is_open = False
        if container is not None:
            try:
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()
        if exc_type is not None:
            return
        if self._count == 0:
            logger.warning(
                "DepthSink wrote no frames; %s has no depth.json.", self.target_dir
            )
            return
        assert self._first is not None
        self._write_manifest(self._first, wrote_preview=container is not None)

    def _write_manifest(self, first: DepthMap, wrote_preview: bool) -> None:
        """Write the clip manifest from the first map and the clip range."""
        manifest = _manifest_header(
            kind=first.kind,
            resolution_wh=first.resolution_wh,
            scale=self._scale,
            camera=first.camera,
            display_range=self.clip_range.display_range,
            view=first.view,
        )
        manifest["frames"] = {"count": self._count, "exact": _EXACT_PATTERN}
        if wrote_preview:
            preview_path = self.target_dir / _PREVIEW_FILE
            preview: dict[str, Any] = {"file": _PREVIEW_FILE}
            codec = avc_codec_string(preview_path)
            if codec is not None:
                preview["codec"] = codec
            preview["levels"] = "tv"
            preview["reserved_max"] = self.reserved_max
            preview["range_px"] = [0, self.clip_range.max_value]
            manifest["preview"] = preview
        manifest_path = self.target_dir / "depth.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
