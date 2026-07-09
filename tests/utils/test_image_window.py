"""Tests for ImageWindow and _bgr_to_pil helper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from supervision.utils.image_window import ImageWindow, _bgr_to_pil, _fit_image


class TestBgrToPil:
    def test_grayscale(self):
        """Grayscale (H,W) array produces an L-mode PIL image of the same size."""
        arr = np.zeros((10, 20), dtype=np.uint8)
        img = _bgr_to_pil(arr)
        assert img.mode == "L"
        assert img.size == (20, 10)

    def test_bgr(self):
        """BGR (H,W,3) array is converted to RGB-mode PIL image."""
        arr = np.zeros((10, 20, 3), dtype=np.uint8)
        arr[:, :, 0] = 10  # B
        arr[:, :, 1] = 20  # G
        arr[:, :, 2] = 30  # R
        img = _bgr_to_pil(arr)
        assert img.mode == "RGB"
        pixel = img.getpixel((0, 0))
        assert pixel == (30, 20, 10)  # R, G, B after swap

    def test_bgra(self):
        """BGRA (H,W,4) array is reordered to RGBA-mode PIL image."""
        arr = np.zeros((10, 20, 4), dtype=np.uint8)
        arr[:, :, 0] = 10  # B
        arr[:, :, 1] = 20  # G
        arr[:, :, 2] = 30  # R
        arr[:, :, 3] = 255  # A
        img = _bgr_to_pil(arr)
        assert img.mode == "RGBA"
        pixel = img.getpixel((0, 0))
        assert pixel == (30, 20, 10, 255)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((10, 20, 2), id="2-channel"),
            pytest.param((10, 20, 5), id="5-channel"),
            pytest.param((10, 20, 3, 1), id="4d"),
        ],
    )
    def test_invalid_shape_raises(self, shape):
        """Unsupported array shapes raise ValueError."""
        arr = np.zeros(shape, dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected shape"):
            _bgr_to_pil(arr)


class TestFitImage:
    def test_scale_down_width_limited(self):
        """Width-constrained image is scaled so width fits exactly."""
        img = Image.new("RGB", (200, 100))
        result = _fit_image(img, 100, 200)
        assert result.size == (100, 50)

    def test_scale_down_height_limited(self):
        """Height-constrained image is scaled so height fits exactly."""
        img = Image.new("RGB", (100, 200))
        result = _fit_image(img, 200, 100)
        assert result.size == (50, 100)

    def test_scale_up(self):
        """Image smaller than the window is scaled up to fill it."""
        img = Image.new("RGB", (50, 50))
        result = _fit_image(img, 200, 200)
        assert result.size == (200, 200)

    def test_same_size_returns_original(self):
        """Image already matching the window is returned unchanged."""
        img = Image.new("RGB", (100, 100))
        result = _fit_image(img, 100, 100)
        assert result is img

    def test_aspect_ratio_preserved_by_default(self):
        """Non-square image keeps its aspect ratio when scaled."""
        img = Image.new("RGB", (400, 200))
        result = _fit_image(img, 100, 100)
        assert result.size == (100, 50)

    def test_free_form_stretches_to_exact_size(self):
        """keep_aspect_ratio=False stretches the image to the exact window size."""
        img = Image.new("RGB", (400, 200))
        result = _fit_image(img, 100, 100, keep_aspect_ratio=False)
        assert result.size == (100, 100)

    def test_free_form_same_size_returns_original(self):
        """keep_aspect_ratio=False with matching size returns the original."""
        img = Image.new("RGB", (100, 100))
        result = _fit_image(img, 100, 100, keep_aspect_ratio=False)
        assert result is img


class TestImageWindowShow:
    @pytest.mark.parametrize(
        ("array", "exc_type", "match"),
        [
            pytest.param(
                np.zeros((10, 10, 3), dtype=np.float32),
                TypeError,
                "uint8",
                id="float32-dtype-raises-type-error",
            ),
            pytest.param(
                np.zeros((10, 10, 2), dtype=np.uint8),
                ValueError,
                "Expected shape",
                id="2-channel-raises-value-error",
            ),
        ],
    )
    def test_show_raises_for_invalid_input(self, array, exc_type, match):
        """show() rejects arrays with invalid dtype or shape."""
        window = ImageWindow("test")
        with pytest.raises(exc_type, match=match):
            window.show(array)

    def test_show_creates_window_and_updates(self):
        """show() creates a Tk window, sets the PhotoImage, and calls update."""
        window = ImageWindow("preview")
        mock_root = MagicMock()
        mock_label = MagicMock()
        mock_photo = MagicMock()
        fake_imagetk = MagicMock()
        fake_imagetk.PhotoImage.return_value = mock_photo

        with (
            patch("supervision.utils.image_window.ImageWindow._ensure_window"),
            patch.dict("sys.modules", {"PIL.ImageTk": fake_imagetk}),
        ):
            window._root = mock_root
            window._label = mock_label

            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            window.show(frame)

        fake_imagetk.PhotoImage.assert_called_once()
        mock_label.configure.assert_called_once_with(image=mock_photo)
        mock_root.update_idletasks.assert_called_once()
        mock_root.update.assert_called_once()


class TestImageWindowUpdateDisplay:
    def test_no_op_without_pil_image(self):
        """_update_display() is a no-op when no image has been shown yet."""
        window = ImageWindow()
        window._label = MagicMock()
        window._update_display()
        window._label.configure.assert_not_called()

    def test_no_op_without_label(self):
        """_update_display() does not raise when the window has no label."""
        window = ImageWindow()
        window._pil_image = Image.new("RGB", (10, 10))
        window._update_display()  # must not raise

    def test_uses_native_size_when_window_size_unknown(self):
        """Native resolution is used when _win_w/_win_h are still 0."""
        window = ImageWindow()
        window._pil_image = Image.new("RGB", (80, 60))
        mock_label = MagicMock()
        window._label = mock_label
        fake_imagetk = MagicMock()

        with patch.dict("sys.modules", {"PIL.ImageTk": fake_imagetk}):
            window._update_display()

        displayed = fake_imagetk.PhotoImage.call_args[0][0]
        assert displayed.size == (80, 60)

    def test_scales_image_to_window_size(self):
        """Image is rescaled to fit _win_w x _win_h when dimensions are known."""
        window = ImageWindow()
        window._pil_image = Image.new("RGB", (400, 200))
        window._win_w = 200
        window._win_h = 200
        window._label = MagicMock()
        fake_imagetk = MagicMock()

        with patch.dict("sys.modules", {"PIL.ImageTk": fake_imagetk}):
            window._update_display()

        # 400x200 into 200x200, width-constrained, gives 200x100
        displayed = fake_imagetk.PhotoImage.call_args[0][0]
        assert displayed.size == (200, 100)

    def test_stretches_image_when_keep_aspect_ratio_false(self):
        """Image is stretched to fill window when keep_aspect_ratio=False."""
        window = ImageWindow(keep_aspect_ratio=False)
        window._pil_image = Image.new("RGB", (400, 200))
        window._win_w = 200
        window._win_h = 200
        window._label = MagicMock()
        fake_imagetk = MagicMock()

        with patch.dict("sys.modules", {"PIL.ImageTk": fake_imagetk}):
            window._update_display()

        displayed = fake_imagetk.PhotoImage.call_args[0][0]
        assert displayed.size == (200, 200)


class TestImageWindowOnConfigure:
    def test_non_root_widget_is_ignored(self):
        """<Configure> events from child widgets do not update dimensions."""
        window = ImageWindow()
        mock_root = MagicMock()
        window._root = mock_root
        event = MagicMock()
        event.widget = MagicMock()  # different object, not root
        event.width = 200
        event.height = 100

        with patch.object(window, "_update_display") as mock_update:
            window._on_configure(event)

        assert window._win_w == 0
        assert window._win_h == 0
        mock_update.assert_not_called()

    def test_same_size_event_is_ignored(self):
        """<Configure> with unchanged dimensions does not trigger a redraw."""
        window = ImageWindow()
        mock_root = MagicMock()
        window._root = mock_root
        window._win_w = 200
        window._win_h = 100
        event = MagicMock()
        event.widget = mock_root
        event.width = 200
        event.height = 100

        with patch.object(window, "_update_display") as mock_update:
            window._on_configure(event)

        mock_update.assert_not_called()

    def test_degenerate_size_is_ignored(self):
        """<Configure> with width or height <= 1 (pre-geometry) is skipped."""
        window = ImageWindow()
        mock_root = MagicMock()
        window._root = mock_root
        event = MagicMock()
        event.widget = mock_root
        event.width = 1
        event.height = 100

        with patch.object(window, "_update_display") as mock_update:
            window._on_configure(event)

        assert window._win_w == 0
        mock_update.assert_not_called()

    def test_new_size_updates_dimensions_and_redraws(self):
        """<Configure> with new dimensions updates _win_w/_win_h and redraws."""
        window = ImageWindow()
        mock_root = MagicMock()
        window._root = mock_root
        window._pil_image = Image.new("RGB", (100, 100))
        event = MagicMock()
        event.widget = mock_root
        event.width = 320
        event.height = 240

        with patch.object(window, "_update_display") as mock_update:
            window._on_configure(event)

        assert window._win_w == 320
        assert window._win_h == 240
        mock_update.assert_called_once()

    def test_new_size_without_image_skips_redraw(self):
        """<Configure> with new dimensions but no image updates dims only."""
        window = ImageWindow()
        mock_root = MagicMock()
        window._root = mock_root
        event = MagicMock()
        event.widget = mock_root
        event.width = 320
        event.height = 240

        with patch.object(window, "_update_display") as mock_update:
            window._on_configure(event)

        assert window._win_w == 320
        assert window._win_h == 240
        mock_update.assert_not_called()


class TestImageWindowWaitKey:
    def test_wait_key_returns_none_when_no_window(self):
        """wait_key() returns None immediately when no window exists."""
        window = ImageWindow()
        assert window.wait_key(delay_ms=0) is None
        assert window.wait_key(delay_ms=1) is None

    def test_wait_key_returns_none_when_closed_mid_wait(self):
        """wait_key(0) returns None when close() fires during blocking wait."""
        window = ImageWindow()
        mock_root = MagicMock()
        mock_event = MagicMock()
        mock_root.wait_variable.side_effect = lambda _: window.close()
        window._root = mock_root
        window._key_event = mock_event

        result = window.wait_key(delay_ms=0)

        assert result is None
        assert window._root is None

    def test_wait_key_returns_queued_key_immediately(self):
        """wait_key() returns the first queued keysym without calling update."""
        window = ImageWindow()
        window._root = MagicMock()
        window._key_queue = ["q", "Escape"]
        result = window.wait_key(delay_ms=1)
        assert result == "q"
        assert window._key_queue == ["Escape"]

    def test_wait_key_blocks_with_tk_event_loop(self):
        """Blocking wait_key() uses Tk wait_variable instead of update polling."""
        window = ImageWindow()
        mock_root = MagicMock()
        mock_event = MagicMock()
        mock_root.wait_variable.side_effect = lambda _: window._key_queue.append("q")
        window._root = mock_root
        window._key_event = mock_event

        result = window.wait_key(delay_ms=0)

        assert result == "q"
        mock_root.wait_variable.assert_called_once_with(mock_event)
        mock_root.update.assert_not_called()

    def test_wait_key_returns_none_on_timeout(self):
        """wait_key() returns None when no key arrives before the deadline."""
        window = ImageWindow()
        mock_root = MagicMock()
        mock_event = MagicMock()
        mock_root.after.return_value = "timeout-id"
        window._root = mock_root
        window._key_event = mock_event

        result = window.wait_key(delay_ms=1)

        assert result is None
        mock_root.after.assert_called_once_with(1, window._signal_wait)
        mock_root.wait_variable.assert_called_once_with(mock_event)
        mock_root.after_cancel.assert_called_once_with("timeout-id")
        mock_root.update.assert_not_called()


class TestImageWindowClose:
    def test_close_destroys_root(self):
        """close() calls destroy() on the Tk root and nulls internal refs."""
        window = ImageWindow()
        mock_root = MagicMock()
        window._root = mock_root
        window._label = MagicMock()
        window._photo = MagicMock()

        window.close()

        mock_root.destroy.assert_called_once()
        assert window._root is None
        assert window._label is None
        assert window._photo is None

    def test_close_signals_waiters_and_clears_key_event(self):
        """close() wakes wait_key() callers and clears stale Tk references."""
        window = ImageWindow()
        mock_root = MagicMock()
        mock_event = MagicMock()
        mock_event.get.return_value = 0
        window._root = mock_root
        window._key_event = mock_event

        window.close()

        mock_event.set.assert_called_once_with(1)
        mock_root.destroy.assert_called_once()
        assert window._root is None
        assert window._key_event is None

    def test_close_is_idempotent(self):
        """close() on an already-closed window does not raise."""
        window = ImageWindow()
        window.close()  # no window created
        window.close()  # second call must not raise

    def test_close_clears_key_queue(self):
        """close() discards stale keys so they don't fire on next open."""
        window = ImageWindow()
        window._root = MagicMock()
        window._key_queue = ["q", "Escape"]
        window.close()
        assert window._key_queue == []

    def test_window_exists_returns_false_for_destroyed_root(self):
        """Destroyed Tk roots are not reused after a window-manager close."""
        window = ImageWindow()
        mock_root = MagicMock()
        mock_root.winfo_exists.return_value = 0
        window._root = mock_root

        assert window._window_exists() is False


class TestImageWindowContextManager:
    def test_context_manager_closes_on_exit(self):
        """The with-statement calls close() when the block exits normally."""
        with patch.object(ImageWindow, "close") as mock_close:
            with ImageWindow("ctx") as w:
                assert isinstance(w, ImageWindow)
            mock_close.assert_called_once()

    def test_context_manager_closes_on_exception(self):
        """close() is called even when the body raises."""
        with patch.object(ImageWindow, "close") as mock_close:
            with pytest.raises(RuntimeError):
                with ImageWindow("ctx"):
                    raise RuntimeError("boom")
            mock_close.assert_called_once()


class TestImageWindowMouseCallback:
    def test_set_mouse_callback_stores_callable(self):
        """set_mouse_callback() stores the callable for later use."""
        window = ImageWindow()
        cb = MagicMock()
        window.set_mouse_callback(cb)
        assert window._mouse_callback is cb

    def test_set_mouse_callback_none_removes_callback(self):
        """Passing None removes a previously set callback."""
        window = ImageWindow()
        window.set_mouse_callback(MagicMock())
        window.set_mouse_callback(None)
        assert window._mouse_callback is None

    def test_on_mouse_forwards_raw_coords_without_image(self):
        """_on_mouse() forwards raw coordinates when no image has been shown."""
        window = ImageWindow()
        cb = MagicMock()
        window._mouse_callback = cb
        event = MagicMock()
        event.x = 5
        event.y = 10
        window._on_mouse(event, "down")
        cb.assert_called_once_with(5, 10, "down")

    def test_on_mouse_maps_scaled_coords_to_image_pixels(self):
        """Event coordinates are unscaled back to original image pixels."""
        window = ImageWindow()
        window._pil_image = Image.new("RGB", (400, 200))
        window._win_w = 200
        window._win_h = 200
        window._update_display_transform(Image.new("RGB", (200, 100)))
        cb = MagicMock()
        window._mouse_callback = cb
        event = MagicMock()
        event.x = 100  # display space -> 200 in original
        event.y = 25  # letterboxed by (200 - 100) / 2 = 50 -> 0 in original
        window._on_mouse(event, "move")
        cb.assert_called_once_with(200, 0, "move")

    def test_on_mouse_clamps_coords_into_image_bounds(self):
        """Coordinates outside the displayed image are clamped to its bounds."""
        window = ImageWindow()
        window._pil_image = Image.new("RGB", (400, 200))
        window._win_w = 200
        window._win_h = 200
        window._update_display_transform(Image.new("RGB", (200, 100)))
        cb = MagicMock()
        window._mouse_callback = cb
        event = MagicMock()
        event.x = 500  # far past the right edge
        event.y = 0  # inside the top letterbox band
        window._on_mouse(event, "down")
        cb.assert_called_once_with(399, 0, "down")

    def test_on_mouse_after_configure_resize_maps_correct_coords(self):
        """End-to-end: a <Configure> resize followed by a click maps to image pixels."""
        window = ImageWindow()
        window._pil_image = Image.new("RGB", (400, 200))
        window._label = MagicMock()
        mock_root = MagicMock()
        window._root = mock_root
        cb = MagicMock()
        window._mouse_callback = cb
        configure_event = MagicMock()
        configure_event.widget = mock_root
        configure_event.width = 200
        configure_event.height = 200
        fake_imagetk = MagicMock()

        with patch.dict("sys.modules", {"PIL.ImageTk": fake_imagetk}):
            window._on_configure(configure_event)  # triggers _update_display internally

        mouse_event = MagicMock()
        mouse_event.x = 100  # display space -> 200 in original
        mouse_event.y = 25  # letterboxed by (200 - 100) / 2 = 50 -> 0 in original
        window._on_mouse(mouse_event, "down")
        cb.assert_called_once_with(200, 0, "down")

    def test_on_mouse_maps_coords_when_stretched(self):
        """Event coords unscale correctly when keep_aspect_ratio=False (stretched)."""
        window = ImageWindow(keep_aspect_ratio=False)
        window._pil_image = Image.new("RGB", (400, 200))
        window._win_w = 100
        window._win_h = 100
        window._update_display_transform(Image.new("RGB", (100, 100)))
        cb = MagicMock()
        window._mouse_callback = cb
        event = MagicMock()
        event.x = 50  # stretched display space -> 200 in original width
        event.y = 50  # stretched display space -> 100 in original height
        window._on_mouse(event, "up")
        cb.assert_called_once_with(200, 100, "up")
