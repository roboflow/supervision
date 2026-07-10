from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from supervision.utils.conversion import cv2_to_pillow

MouseCallback = Callable[[int, int, str], None]


class ImageWindow:
    """Desktop image window backed by stdlib tkinter + pillow.

    Functional replacement for `cv2.imshow` / `cv2.waitKey` that works under
    `opencv-python-headless`. Requires tkinter (stdlib) and pillow (already a
    supervision dependency). Instantiation always succeeds; `show()` raises
    `ModuleNotFoundError` (No module named '_tkinter') on environments where
    `python3-tk` is absent — install with ``sudo apt-get install python3-tk``
    (Debian/Ubuntu) or ``brew install tcl-tk`` (macOS with Homebrew/pyenv)
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

    Args:
        title: Window title bar text.
        keep_aspect_ratio: If ``True`` (default), frames are scaled to fit
            inside the window while preserving aspect ratio (letterboxed). If
            ``False``, frames are stretched to fill the window exactly.

    Examples:
        ```python
        import supervision as sv

        window = sv.ImageWindow("preview")
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

        with sv.ImageWindow("preview") as window:
            for frame in sv.get_video_frames_generator(source_path="video.mp4"):
                window.show(frame)
                if window.wait_key(delay_ms=1) == "q":
                    break
        ```
    """

    def __init__(
        self, title: str = "supervision", keep_aspect_ratio: bool = True
    ) -> None:
        self.title = title
        self.keep_aspect_ratio = keep_aspect_ratio
        self._mouse_callback: MouseCallback | None = None
        self._root: Any = None
        self._label: Any = None
        self._photo: Any = None
        self._key_event: Any = None
        self._key_queue: list[str] = []
        self._pil_image: Image.Image | None = None
        self._win_w: int = 0
        self._win_h: int = 0
        self._display_scale_x: float = 1.0
        self._display_scale_y: float = 1.0
        self._display_offset_x: float = 0.0
        self._display_offset_y: float = 0.0

    def show(self, image: npt.NDArray[np.uint8]) -> None:
        """Display a BGR, grayscale, or BGRA frame in the window.

        The image is scaled to fit the current window dimensions. Aspect ratio is
        preserved unless `keep_aspect_ratio=False`. Resizing the window rescales live.
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
        self._pil_image = _bgr_to_pil(image)
        self._ensure_window()
        self._update_display()
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

    @property
    def is_open(self) -> bool:
        """Return whether the underlying Tk window currently exists."""
        return self._window_exists()

    def close(self) -> None:
        """Destroy the window and release its resources."""
        if self._root is not None:
            root = self._root
            self._signal_wait()
            with suppress(Exception):
                root.destroy()
            self._reset_window_refs()

    def __enter__(self) -> ImageWindow:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_window(self) -> None:
        if self._window_exists():
            return
        self._reset_window_refs()
        import tkinter as tk

        self._root = tk.Tk()
        self._root.title(self.title)
        self._label = tk.Label(self._root)
        self._label.pack(expand=True, fill="both")
        self._key_event = tk.IntVar(master=self._root, value=0)
        self._root.bind("<Key>", self._on_key)
        self._root.bind("<Configure>", self._on_configure)
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._label.bind("<Button-1>", lambda e: self._on_mouse(e, "down"))
        self._label.bind("<ButtonRelease-1>", lambda e: self._on_mouse(e, "up"))
        self._label.bind("<Motion>", lambda e: self._on_mouse(e, "move"))

    def _on_configure(self, event: Any) -> None:
        if event.widget is not self._root:
            return
        w, h = event.width, event.height
        if w > 1 and h > 1 and (w, h) != (self._win_w, self._win_h):
            self._win_w, self._win_h = w, h
            if self._pil_image is not None:
                self._update_display()

    def _update_display(self) -> None:
        if self._pil_image is None or self._label is None:
            return
        from PIL import ImageTk

        img = self._pil_image
        if self._win_w > 1 and self._win_h > 1:
            img = _fit_image(img, self._win_w, self._win_h, self.keep_aspect_ratio)
        self._update_display_transform(img)
        self._photo = ImageTk.PhotoImage(img)
        self._label.configure(image=self._photo)

    def _update_display_transform(self, fitted: Image.Image) -> None:
        """Record the scale and letterbox offset from the last display pass.

        The fitted image is drawn centered inside the label, so mouse events on
        the label carry display-space coordinates. Storing the per-axis scale
        and centering offset lets `_on_mouse` invert the transform back to
        original image-pixel coordinates. Falls back to an identity transform
        while the window geometry is still unknown.
        """
        if self._pil_image is None:
            return
        orig_w, orig_h = self._pil_image.size
        fit_w, fit_h = fitted.size
        win_w = self._win_w if self._win_w > 1 else fit_w
        win_h = self._win_h if self._win_h > 1 else fit_h
        self._display_scale_x = fit_w / orig_w if orig_w else 1.0
        self._display_scale_y = fit_h / orig_h if orig_h else 1.0
        self._display_offset_x = max(0.0, (win_w - fit_w) / 2)
        self._display_offset_y = max(0.0, (win_h - fit_h) / 2)

    def _on_key(self, event: Any) -> None:
        self._key_queue.append(event.keysym)
        self._signal_wait()

    def _on_mouse(self, event: Any, event_type: str) -> None:
        if self._mouse_callback is None:
            return
        x, y = self._to_image_coords(event.x, event.y)
        self._mouse_callback(x, y, event_type)

    def _to_image_coords(self, event_x: int, event_y: int) -> tuple[int, int]:
        """Map label-space event coordinates to original image pixels.

        Inverts the scale and letterbox offset recorded by `_update_display`
        so callbacks receive coordinates in the source image's pixel space,
        regardless of how the window has been resized. Results are rounded to
        the nearest pixel and clamped inside the image bounds. When no image
        has been shown, the raw event coordinates are returned unchanged.
        """
        if self._pil_image is None:
            return event_x, event_y
        orig_w, orig_h = self._pil_image.size
        image_x = round((event_x - self._display_offset_x) / self._display_scale_x)
        image_y = round((event_y - self._display_offset_y) / self._display_scale_y)
        clamped_x = min(max(image_x, 0), orig_w - 1)
        clamped_y = min(max(image_y, 0), orig_h - 1)
        return clamped_x, clamped_y

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


def _fit_image(
    image: Image.Image, width: int, height: int, keep_aspect_ratio: bool = True
) -> Image.Image:
    iw, ih = image.size
    if keep_aspect_ratio:
        scale = min(width / iw, height / ih)
        new_w, new_h = max(1, round(iw * scale)), max(1, round(ih * scale))
    else:
        new_w, new_h = width, height
    if (new_w, new_h) == (iw, ih):
        return image
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _bgr_to_pil(image: npt.NDArray[np.uint8]) -> Image.Image:
    """Convert a BGR, grayscale, or BGRA OpenCV array to a Pillow image.

    Thin wrapper delegating to `sv.cv2_to_pillow`, which reorders channels
    from OpenCV's BGR(A) convention to Pillow's RGB(A) and passes grayscale
    arrays through unchanged.
    """
    return cv2_to_pillow(image)
