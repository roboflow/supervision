"""Tests for TkImageWindow and _bgr_to_pil helper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from supervision.utils.image_window import TkImageWindow, _bgr_to_pil


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


class TestTkImageWindowShow:
    def test_show_raises_type_error_for_non_uint8(self):
        """show() rejects arrays whose dtype is not uint8."""
        window = TkImageWindow("test")
        with pytest.raises(TypeError, match="uint8"):
            window.show(np.zeros((10, 10, 3), dtype=np.float32))

    def test_show_raises_value_error_for_bad_shape(self):
        """show() propagates ValueError for unsupported array shapes."""
        window = TkImageWindow("test")
        with pytest.raises(ValueError, match="Expected shape"):
            window.show(np.zeros((10, 10, 2), dtype=np.uint8))

    def test_show_creates_window_and_updates(self):
        """show() creates a Tk window, sets the PhotoImage, and calls update."""
        window = TkImageWindow("preview")
        mock_root = MagicMock()
        mock_label = MagicMock()
        mock_photo = MagicMock()

        with (
            patch("supervision.utils.image_window.TkImageWindow._ensure_window"),
            patch("PIL.ImageTk.PhotoImage", return_value=mock_photo) as mock_ph,
        ):
            window._root = mock_root
            window._label = mock_label

            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            window.show(frame)

        mock_ph.assert_called_once()
        mock_label.configure.assert_called_once_with(image=mock_photo)
        mock_root.update_idletasks.assert_called_once()
        mock_root.update.assert_called_once()


class TestTkImageWindowWaitKey:
    def test_wait_key_returns_queued_key_immediately(self):
        """wait_key() returns the first queued keysym without calling update."""
        window = TkImageWindow()
        window._root = MagicMock()
        window._key_queue = ["q", "Escape"]
        result = window.wait_key(delay_ms=1)
        assert result == "q"
        assert window._key_queue == ["Escape"]

    def test_wait_key_blocks_with_tk_event_loop(self):
        """Blocking wait_key() uses Tk wait_variable instead of update polling."""
        window = TkImageWindow()
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
        window = TkImageWindow()
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


class TestTkImageWindowClose:
    def test_close_destroys_root(self):
        """close() calls destroy() on the Tk root and nulls internal refs."""
        window = TkImageWindow()
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
        window = TkImageWindow()
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
        window = TkImageWindow()
        window.close()  # no window created
        window.close()  # second call must not raise

    def test_window_exists_returns_false_for_destroyed_root(self):
        """Destroyed Tk roots are not reused after a window-manager close."""
        window = TkImageWindow()
        mock_root = MagicMock()
        mock_root.winfo_exists.return_value = 0
        window._root = mock_root

        assert window._window_exists() is False


class TestTkImageWindowContextManager:
    def test_context_manager_closes_on_exit(self):
        """The with-statement calls close() when the block exits normally."""
        with patch.object(TkImageWindow, "close") as mock_close:
            with TkImageWindow("ctx") as w:
                assert isinstance(w, TkImageWindow)
            mock_close.assert_called_once()

    def test_context_manager_closes_on_exception(self):
        """close() is called even when the body raises."""
        with patch.object(TkImageWindow, "close") as mock_close:
            with pytest.raises(RuntimeError):
                with TkImageWindow("ctx"):
                    raise RuntimeError("boom")
            mock_close.assert_called_once()


class TestTkImageWindowMouseCallback:
    def test_set_mouse_callback_stores_callable(self):
        """set_mouse_callback() stores the callable for later use."""
        window = TkImageWindow()
        cb = MagicMock()
        window.set_mouse_callback(cb)
        assert window._mouse_callback is cb

    def test_set_mouse_callback_none_removes_callback(self):
        """Passing None removes a previously set callback."""
        window = TkImageWindow()
        window.set_mouse_callback(MagicMock())
        window.set_mouse_callback(None)
        assert window._mouse_callback is None

    def test_on_mouse_calls_callback_with_coords(self):
        """_on_mouse() forwards (x, y, event_type) to the registered callback."""
        window = TkImageWindow()
        cb = MagicMock()
        window._mouse_callback = cb
        event = MagicMock()
        event.x = 5
        event.y = 10
        window._on_mouse(event, "down")
        cb.assert_called_once_with(5, 10, "down")
