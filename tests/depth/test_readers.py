from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import supervision as sv
from supervision.depth.readers import read_pfm


class TestReadPfm:
    @staticmethod
    def _write(path: Path, values: np.ndarray, little_endian: bool) -> None:
        """Write a grayscale PFM, rows bottom to top, in the given byte order."""
        height, width = values.shape
        scale = -1.0 if little_endian else 1.0
        dtype = "<f4" if little_endian else ">f4"
        header = f"Pf\n{width} {height}\n{scale}\n".encode()
        path.write_bytes(header + values[::-1].astype(dtype).tobytes())

    @pytest.mark.parametrize("little_endian", [True, False])
    def test_reads_rows_top_first_in_either_byte_order(
        self, tmp_path: Path, little_endian: bool
    ) -> None:
        """Rows are flipped and the scale sign picks the byte order."""
        values = np.array([[1.0, 2.0, np.inf], [3.0, 4.0, 5.5]], dtype=np.float32)
        self._write(tmp_path / "disp0.pfm", values, little_endian)

        loaded = read_pfm(tmp_path / "disp0.pfm")

        np.testing.assert_array_equal(loaded, values)

    def test_infinite_disparity_is_no_depth(self, tmp_path: Path) -> None:
        """Middlebury's +inf for unknown disparity reads as no depth."""
        values = np.array([[np.inf, 4.0]], dtype=np.float32)
        self._write(tmp_path / "disp0.pfm", values, little_endian=True)

        depth_map = sv.DepthMap.from_pfm(tmp_path / "disp0.pfm")

        assert depth_map.valid_mask.tolist() == [[False, True]]

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            pytest.param(b"PF\n1 1\n-1.0\n" + b"\0" * 12, "colour", id="colour"),
            pytest.param(b"P6\n1 1\n255\n\0\0\0", "not a PFM", id="ppm"),
            pytest.param(b"Pf\n2 2\n-1.0\n\0\0\0\0", "truncated", id="truncated"),
        ],
    )
    def test_rejects_other_files(
        self, tmp_path: Path, content: bytes, match: str
    ) -> None:
        """Only complete grayscale PFMs are depth maps."""
        (tmp_path / "file.pfm").write_bytes(content)

        with pytest.raises(ValueError, match=match):
            read_pfm(tmp_path / "file.pfm")
