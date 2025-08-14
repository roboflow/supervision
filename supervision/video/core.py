from __future__ import annotations

import os
import sys
from collections.abc import Callable

import cv2
import numpy as np
from tqdm.auto import tqdm

from supervision.video.backend import Backend, BackendDict, BackendTypes, WriterTypes
from supervision.video.utils import SourceType, VideoInfo

try:
    import IPython.display as iPyDisplay
except ImportError:
    iPyDisplay = None


class Video:
    """
    A high-level interface for reading, processing, and writing video files or streams.

    Attributes:
        info (VideoInfo): Metadata about the opened video.
        source (str | int): Path to the video file or index of the camera device.
        backend (BackendTypes): Video backend used for I/O operations.
    """

    info: VideoInfo
    source: str | int
    backend: BackendTypes

    def __init__(
        self, source: str | int, backend: Backend | str = Backend.OPENCV
    ) -> None:
        """
        Initialize the Video object and open the source.

        Args:
            source (str | int): Path to a video file or index of a camera device.
            backend (Backend | str, optional): Backend type or name for video I/O.
                Defaults to Backend.OPENCV.

        Raises:
            ValueError: If the specified backend is not supported.
        """
        self.backend = BackendDict.get(Backend.from_value(backend))
        if self.backend is None:
            raise ValueError(f"Unsupported backend: {backend}")

        # Instantiate the backend class once sanity check is done
        self.backend = self.backend()

        self.backend.open(source)
        self.info = self.backend.info()
        self.source = source

    def __iter__(self):
        """
        Make the Video object directly iterable over frames.

        Yields:
            np.ndarray: The next frame in the video stream.
        """
        return self.backend.frames()

    def sink(
        self,
        target_path: str,
        info: VideoInfo,
        codec: str | None = None,
        render_audio: bool | None = None,
    ) -> WriterTypes:
        """
        Create a video writer for saving frames to a file.

        Args:
            target_path (str): Output file path for the video.
            info (VideoInfo): Video metadata including resolution and FPS.
            codec (str, optional): FourCC video codec code.
                If None, the backend's default codec is used.
            render_audio (bool | None, optional): Whether to include audio if supported.

        Returns:
            WriterTypes: Video writer instance for writing frames.
        """
        return self.backend.writer(
            target_path, info.fps, info.resolution_wh, codec, self.backend, render_audio
        )

    def frames(
        self,
        stride: int = 1,
        start: int = 0,
        end: int | None = None,
        resolution_wh: tuple[int, int] | None = None,
    ):
        """
        Generate frames from the video with optional skipping, seeking, and resizing.

        Args:
            stride (int, optional): Number of frames to skip between each yield.
                Defaults to 1 (no skipping).
            start (int, optional): Index of the first frame to read. Defaults to 0.
            end (int | None, optional): Index after the last frame to read.
                If None, reads until the end of the video.
            resolution_wh (tuple[int, int] | None, optional): Target resolution
                (width, height) for resizing frames. If None, keeps original size.

        Yields:
            np.ndarray: The next frame in the video.

        Raises:
            RuntimeError: If the video has not been opened.
        """
        if self.backend.cap is None:
            raise RuntimeError("Video not opened yet.")

        total_frames = (
            self.backend.video_info.total_frames if self.backend.video_info else 0
        )
        is_live_stream = total_frames is None or total_frames <= 0

        if is_live_stream:
            # Live stream handling
            while True:
                for _ in range(stride - 1):
                    if not self.backend.grab():
                        return
                ret, frame = self.backend.read()
                if not ret:
                    return
                if resolution_wh is not None:
                    frame = cv2.resize(frame, resolution_wh)
                yield frame
        else:
            # Video file handling
            if end is None or end > total_frames:
                end = total_frames

            frame_idx = start
            while frame_idx < end:
                self.backend.seek(frame_idx)
                ret, frame = self.backend.read()
                if not ret:
                    break
                if resolution_wh is not None:
                    frame = cv2.resize(frame, resolution_wh)
                yield frame
                frame_idx += stride

    def save(
        self,
        target_path: str,
        callback: Callable[[np.ndarray, int], np.ndarray],
        fps: int | None = None,
        progress_message: str = "Processing video",
        show_progress: bool = False,
        codec: str | None = None,
        render_audio: bool | None = None,
    ):
        """
        Process and save video frames to a file.

        Reads frames from the source, applies the given `callback` function to each
        frame, and writes the processed frames to the specified output file.

        Args:
            target_path (str): Output file path for the processed video.
            callback (Callable[[np.ndarray, int], np.ndarray]): A function that takes in
                a video frame (numpy array) and its frame index, and returns a frame.
            fps (int | None, optional): Frames per second of the output video.
                If None, uses the original FPS.
            progress_message (str, optional): Message displayed in the progress bar.
                Defaults to "Processing video".
            show_progress (bool, optional): If True, displays a tqdm progress bar.
                Defaults to False.
            codec (str | None, optional): FourCC video codec code.
                If None, uses the backend's default codec.
            render_audio (bool | None, optional): Whether to include audio if supported.

        Raises:
            RuntimeError: If the video has not been opened.
            ValueError: If the video source is not a file.

        Returns:
            None
        """
        if self.backend.cap is None:
            raise RuntimeError("Video not opened yet.")

        if self.backend.video_info.SourceType != SourceType.VIDEO_FILE:
            raise ValueError("Only video files can be saved.")

        if fps is None:
            fps = self.backend.video_info.fps

        writer = self.backend.writer(
            target_path,
            fps,
            self.backend.video_info.resolution_wh,
            codec,
            self.backend,
            render_audio,
        )
        total_frames = self.backend.video_info.total_frames
        frames_generator = self.frames()
        for index, frame in enumerate(
            tqdm(
                frames_generator,
                total=total_frames,
                disable=not show_progress,
                desc=progress_message,
            )
        ):
            result_frame = callback(frame, index)
            writer.write(frame=result_frame)

        writer.close()

    def show(self, resolution_wh: tuple[int, int] | None = None, 
        callback: Callable[[np.ndarray, int], np.ndarray] = lambda f, i: f,
        fps: int | None = None,
        progress_message: str = "Processing video",
        show_progress: bool = False,
        render_audio: bool | None = None):
        """
        Display video frames in a window with interactive playback controls.

        This method streams video frames to an OpenCV window, allowing real-time
        visualization. Press 'q' to quit playback. The method handles various
        display-related exceptions gracefully.

        Args:
            resolution_wh (tuple[int, int] | None): Optional target resolution as
                (width, height) tuple. If None, uses native video resolution.
                Note: Aspect ratio may not be preserved.
        """

        # On Jupyter Notebook
        def in_notebook():
            argv = getattr(sys, "argv", [])
            return any("jupyter" in arg or "ipykernel_launcher" in arg for arg in argv)

        def is_Headless():
            if sys.platform.startswith("linux"):
                return not bool(os.environ.get("DISPLAY", ""))
            if sys.platform == "darwin":
                return not bool(
                    os.environ.get("TERM_PROGRAM") or os.environ.get("DISPLAY")
                )
            if sys.platform.startswith("win"):
                try:
                    import ctypes

                    user32 = ctypes.windll.user32
                    return user32.GetDesktopWindow() == 0
                except Exception:
                    return True
            return True

        # On a notebook
        if in_notebook():
            if iPyDisplay is None:
                raise RuntimeError(
                    "IPython (`IPython` module) is not installed. "
                    "Run `pip install IPython`."
                )

            self.save("temp.mp4",
                      callback=callback,
                      fps=fps,
                      progress_message=progress_message,
                      show_progress=show_progress,
                      render_audio=render_audio
                      )

            width = resolution_wh[0] if resolution_wh is not None else None
            height = resolution_wh[1] if resolution_wh is not None else None
            iPyDisplay.display(
                iPyDisplay.Video("temp.mp4", embed=True, width=width, height=height)
            )
            os.remove("temp.mp4")
        # On a computer
        elif not is_Headless():
            for frame in self.frames(resolution_wh=resolution_wh):
                cv2.imshow(str(self.source), frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            while True:
                if cv2.getWindowProperty(str(self.source), cv2.WND_PROP_VISIBLE) < 1:
                    break
                cv2.waitKey(100)
            cv2.destroyAllWindows()
        # On a headless system
        else:
            if iPyDisplay is None:
                raise RuntimeError(
                    "IPython (`IPython` module) is not installed. "
                    "Run `pip install IPython`."
                )

            self.save("temp.mp4",
                      callback=callback,
                      fps=fps,
                      progress_message=progress_message,
                      show_progress=show_progress,
                      render_audio=render_audio
                      )

            width = resolution_wh[0] if resolution_wh is not None else None
            height = resolution_wh[1] if resolution_wh is not None else None

            display_video = iPyDisplay.Video(
                "temp.mp4", embed=True, width=width, height=height
            )
            html_code = display_video._repr_html_()
            export_path = "video_display.html"

            with open(export_path, "w") as f:
                f.write(html_code)
            print(f"Video exported as HTML to {export_path}")

            os.remove("temp.mp4")
