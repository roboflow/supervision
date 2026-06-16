from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

MouseCallback = Callable[[int, int, str], None]


class TkImageWindow:
    """Desktop image window backed by stdlib tkinter + pillow.

    Functional replacement for `cv2.imshow` / `cv2.waitKey` that works under
    `opencv-python-headless`. Requires tkinter (stdlib) and pillow (already a
    supervision dependency). Instantiation always succeeds; `show()` raises
    `ModuleNotFoundError` (No module named '_tkinter') on environments where
    `python3-tk` is absent — install with ``sudo apt-get install python3-tk``
    (Debian/Ubuntu) or ``brew install python-tk`` (macOS with Homebrew/pyenv)
    — and raises `tkinter.TclError` on headless servers where a display is
    unavailable.

    Differences from cv2:
        - `wait_key()` returns a tkinter keysym `str` (e.g. ``"q"``,
          ``"Return"``, ``"Escape"``) or ``None``, not an ``int``. Code
          relying on ``key == ord("q")`` or ``key & 0xFF == 27`` must be
          updated to ``key == "q"`` or ``key == "Escape"``.
        - Mouse callback signature is ``(x: int, y: int, event_type: str)``
          where ``event_type`` is ``"down"``, ``"up"``, or ``"move"``. This
          differs from cv2's ``(event: int, x, y, flags, param)`` — existing
          cv2 mouse callbacks are not compatible without modification.
        - Only left-button events are captured. Right-button clicks, scroll
          events, and modifier-key flags (Ctrl, Shift) have no equivalent.

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
        self._key_event: Any = None
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
        if self._root is None and not self._key_queue:
            return None
        self._ensure_window()
        if self._key_queue:
            return self._key_queue.pop(0)
        root = self._root
        if delay_ms <= 0:
            self._wait_for_key_or_close()
        else:
            timeout_id = root.after(delay_ms, self._signal_wait)
            self._wait_for_key_or_close()
            if self._root is not None:
                with suppress(Exception):
                    root.after_cancel(timeout_id)
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
            root = self._root
            self._signal_wait()
            with suppress(Exception):
                root.destroy()
            self._reset_window_refs()

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
        if self._window_exists():
            return
        self._reset_window_refs()
        import tkinter as tk

        self._root = tk.Tk()
        self._root.title(self.title)
        self._label = tk.Label(self._root)
        self._label.pack()
        self._key_event = tk.IntVar(master=self._root, value=0)
        self._root.bind("<Key>", self._on_key)
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._label.bind("<Button-1>", lambda e: self._on_mouse(e, "down"))
        self._label.bind("<ButtonRelease-1>", lambda e: self._on_mouse(e, "up"))
        self._label.bind("<Motion>", lambda e: self._on_mouse(e, "move"))

    def _on_key(self, event: Any) -> None:
        self._key_queue.append(event.keysym)
        self._signal_wait()

    def _on_mouse(self, event: Any, event_type: str) -> None:
        if self._mouse_callback is not None:
            self._mouse_callback(event.x, event.y, event_type)

    def _signal_wait(self) -> None:
        if self._key_event is None:
            return
        with suppress(Exception):
            self._key_event.set(self._key_event.get() + 1)

    def _wait_for_key_or_close(self) -> None:
        if self._root is None or self._key_event is None:
            return
        if self._key_queue:
            return
        try:
            self._root.wait_variable(self._key_event)
        except Exception:
            self._reset_window_refs()

    def _window_exists(self) -> bool:
        if self._root is None:
            return False
        try:
            return bool(self._root.winfo_exists())
        except Exception:
            return False

    def _reset_window_refs(self) -> None:
        self._root = None
        self._label = None
        self._photo = None
        self._key_event = None
        self._key_queue.clear()


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
