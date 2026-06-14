from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

MouseCallback = Callable[[int, int, str], None]


class TkImageWindow:
    """Desktop image window backed by stdlib tkinter + pillow.

    Drop-in replacement for `cv2.imshow` / `cv2.waitKey` that works under
    `opencv-python-headless`. Requires tkinter (stdlib) and pillow (already a
    supervision dependency). On headless servers with no display, instantiation
    succeeds but `show()` / `wait_key()` will raise `tkinter.TclError` when
    the window is first created.

    Attributes:
        title: Window title bar text.

    Examples:
        ```python
        import supervision as sv

        window = sv.TkImageWindow("preview")
        for frame in sv.get_video_frames_generator(source_path="video.mp4"):
            annotated = ...  # annotate frame
            window.show(annotated)
            if window.wait_key(delay_ms=1) == "q":
                break
        window.close()
        ```

        Context-manager form closes automatically:

        ```python
        import supervision as sv

        with sv.TkImageWindow("preview") as window:
            for frame in sv.get_video_frames_generator(source_path="video.mp4"):
                window.show(frame)
                if window.wait_key(delay_ms=1) == "q":
                    break
        ```
    """

    def __init__(self, title: str = "supervision") -> None:
        self.title = title
        self._mouse_callback: MouseCallback | None = None
        self._root: Any = None
        self._label: Any = None
        self._photo: Any = None
        self._key_queue: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self, image: npt.NDArray[np.uint8]) -> None:
        """Display a BGR, grayscale, or BGRA frame in the window.

        Args:
            image: uint8 numpy array. Accepted shapes:
                - ``(H, W)`` — grayscale
                - ``(H, W, 3)`` — BGR (OpenCV convention; channels are swapped
                  to RGB before display)
                - ``(H, W, 4)`` — BGRA (channels reordered to RGBA)

        Raises:
            TypeError: If `image.dtype` is not `uint8`.
            ValueError: If `image` is not 2-D or 3-D with 3 or 4 channels.
        """
        if image.dtype != np.uint8:
            raise TypeError(
                f"image must be uint8, got {image.dtype}. "
                "Convert with image.astype(np.uint8) before calling show()."
            )
        pil_image = _bgr_to_pil(image)
        self._ensure_window()
        from PIL import ImageTk

        self._photo = ImageTk.PhotoImage(pil_image)
        self._label.configure(image=self._photo)
        self._root.update_idletasks()
        self._root.update()

    def wait_key(self, delay_ms: int = 0) -> str | None:
        """Wait for a keypress and return its name.

        Args:
            delay_ms: How long to wait in milliseconds. ``0`` blocks until a
                key is pressed. Positive values poll for up to `delay_ms` ms
                and return ``None`` if no key arrives in time.

        Returns:
            The tkinter keysym string (e.g. ``"q"``, ``"Return"``, ``"Escape"``)
            or ``None`` if the timeout elapsed without a key event.
        """
        self._ensure_window()
        if self._key_queue:
            return self._key_queue.pop(0)
        if delay_ms <= 0:
            while not self._key_queue:
                self._root.update()
        else:
            deadline = time.monotonic() + delay_ms / 1000.0
            while not self._key_queue and time.monotonic() < deadline:
                self._root.update()
        return self._key_queue.pop(0) if self._key_queue else None

    def set_mouse_callback(self, callback: MouseCallback | None) -> None:
        """Register a callback for mouse events on the image.

        Args:
            callback: Callable receiving ``(x, y, event_type)`` where
                ``event_type`` is one of ``"down"``, ``"up"``, or ``"move"``.
                Pass ``None`` to remove the callback.
        """
        self._mouse_callback = callback

    def close(self) -> None:
        """Destroy the window and release its resources."""
        if self._root is not None:
            self._root.destroy()
            self._root = None
            self._label = None
            self._photo = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> TkImageWindow:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_window(self) -> None:
        if self._root is not None:
            return
        import tkinter as tk

        self._root = tk.Tk()
        self._root.title(self.title)
        self._label = tk.Label(self._root)
        self._label.pack()
        self._root.bind("<Key>", self._on_key)
        self._label.bind("<Button-1>", lambda e: self._on_mouse(e, "down"))
        self._label.bind("<ButtonRelease-1>", lambda e: self._on_mouse(e, "up"))
        self._label.bind("<Motion>", lambda e: self._on_mouse(e, "move"))

    def _on_key(self, event: Any) -> None:
        self._key_queue.append(event.keysym)

    def _on_mouse(self, event: Any, event_type: str) -> None:
        if self._mouse_callback is not None:
            self._mouse_callback(event.x, event.y, event_type)


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------


def _bgr_to_pil(image: npt.NDArray[np.uint8]) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(np.ascontiguousarray(image))
    if image.ndim == 3 and image.shape[2] == 3:
        return Image.fromarray(np.ascontiguousarray(image[..., ::-1]))
    if image.ndim == 3 and image.shape[2] == 4:
        return Image.fromarray(np.ascontiguousarray(image[..., [2, 1, 0, 3]]))
    raise ValueError(f"Expected shape (H,W), (H,W,3), or (H,W,4), got {image.shape}.")
