import numpy as np
import pytest

import supervision as sv
from supervision.key_points.angles import JointAngle

VERTEX = [50.0, 50.0]
JOINT = JointAngle(name="joint", a=0, b=1, c=2)


def _single_joint(
    a: list[float], c: list[float], visible: list[list[bool]] | None = None
) -> sv.KeyPoints:
    """Build one skeleton whose points 0, 1 and 2 form the rays b -> a and b -> c."""
    visibility = None if visible is None else np.array(visible)
    xy = np.array([[a, VERTEX, c]], dtype=np.float32)
    return sv.KeyPoints(xy=xy, visible=visibility)


class TestJointAngle:
    @pytest.mark.parametrize(
        ("a", "b", "c"),
        [
            pytest.param(-1, 1, 2, id="negative-index"),
            pytest.param(0, 0, 2, id="vertex-equals-a"),
            pytest.param(0, 1, 1, id="vertex-equals-c"),
            pytest.param(2, 1, 2, id="a-equals-c"),
        ],
    )
    def test_invalid_positions_raise(self, a: int, b: int, c: int) -> None:
        """A joint needs three distinct, non-negative key point positions."""
        with pytest.raises(ValueError, match="JointAngle 'joint'"):
            JointAngle(name="joint", a=a, b=b, c=c)

    @pytest.mark.parametrize(
        "position",
        [
            pytest.param(2.0, id="float"),
            pytest.param(True, id="bool"),
            pytest.param("2", id="string"),
        ],
    )
    def test_non_integer_position_raises(self, position: object) -> None:
        """Positions must be integers; floats, bools and strings are rejected."""
        with pytest.raises(TypeError, match="positions must be integers"):
            JointAngle(name="joint", a=0, b=1, c=position)  # type: ignore[arg-type]

    def test_numpy_integer_positions_are_accepted(self) -> None:
        """NumPy integers, as produced by array indexing, are valid positions."""
        joint = JointAngle(name="joint", a=np.int64(0), b=np.int64(1), c=np.int64(2))

        assert (joint.a, joint.b, joint.c) == (0, 1, 2)


class TestJointAngles:
    @pytest.mark.parametrize(
        ("a", "c", "expected"),
        [
            pytest.param([90.0, 50.0], [50.0, 10.0], 90.0, id="right-angle"),
            pytest.param([90.0, 50.0], [10.0, 50.0], 180.0, id="collinear"),
            pytest.param([90.0, 50.0], [90.0, 10.0], 45.0, id="acute"),
            pytest.param([90.0, 50.0], [10.0, 10.0], 135.0, id="obtuse"),
            pytest.param([90.0, 50.0], [70.0, 50.0], 0.0, id="folded"),
            pytest.param([90.0, 50.0], [90.0, 49.0], 1.432096, id="small-acute"),
        ],
    )
    def test_returns_interior_angle_in_degrees(
        self, a: list[float], c: list[float], expected: float
    ) -> None:
        """The angle at b between the rays to a and c is returned in degrees."""
        key_points = _single_joint(a, c)

        angles = sv.joint_angles(key_points, [JOINT])

        np.testing.assert_allclose(angles, [[expected]], atol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
    def test_result_is_float64(self, dtype: type) -> None:
        """Angles are returned as float64 whatever the key point dtype."""
        xy: np.ndarray = np.array([[[90, 50], [50, 50], [50, 10]]], dtype=dtype)
        key_points = sv.KeyPoints(xy=xy)

        angles = sv.joint_angles(key_points, [JOINT])

        assert angles.dtype == np.float64
        np.testing.assert_allclose(angles, [[90.0]], atol=1e-5)

    def test_returns_one_column_per_joint_and_one_row_per_skeleton(self) -> None:
        """Rows follow the skeletons and columns follow the requested joints."""
        key_points = sv.KeyPoints(
            xy=np.array(
                [
                    [[90, 50], [50, 50], [50, 10], [10, 50]],
                    [[60, 40], [40, 40], [40, 20], [40, 60]],
                ],
                dtype=np.float32,
            )
        )
        joints = [
            JointAngle(name="first", a=0, b=1, c=2),
            JointAngle(name="second", a=0, b=1, c=3),
        ]

        angles = sv.joint_angles(key_points, joints)

        np.testing.assert_allclose(angles, [[90, 180], [90, 90]], atol=1e-5)

    @pytest.mark.parametrize(
        ("a", "visible"),
        [
            pytest.param([np.nan, 50.0], None, id="nan"),
            pytest.param([np.inf, 50.0], None, id="inf"),
            pytest.param([0.0, 0.0], None, id="zero-marker"),
            pytest.param([90.0, 50.0], [[False, True, True]], id="hidden"),
            pytest.param([50.0, 50.0], None, id="zero-length-segment"),
        ],
    )
    def test_unusable_point_gives_nan(
        self, a: list[float], visible: list[list[bool]] | None
    ) -> None:
        """A missing, hidden or degenerate point yields NaN instead of an angle."""
        key_points = _single_joint(a, [50.0, 10.0], visible=visible)

        angles = sv.joint_angles(key_points, [JOINT])

        assert np.isnan(angles).all()

    def test_point_next_to_zero_marker_is_measured(self) -> None:
        """Only exactly [0, 0] marks a missing point; a point just beside it counts."""
        xy = np.array([[[0.001, 0.0], [0.001, 50.0], [40.001, 50.0]]], dtype=np.float32)
        key_points = sv.KeyPoints(xy=xy)

        angles = sv.joint_angles(key_points, [JOINT])

        np.testing.assert_allclose(angles, [[90.0]], atol=1e-5)

    def test_unusable_point_does_not_affect_other_skeletons(self) -> None:
        """Only the skeleton with the missing point gets NaN."""
        key_points = sv.KeyPoints(
            xy=np.array(
                [
                    [[90, 50], [50, 50], [50, 10]],
                    [[90, 50], [0, 0], [50, 10]],
                ],
                dtype=np.float32,
            )
        )

        angles = sv.joint_angles(key_points, [JOINT])

        np.testing.assert_allclose(angles, [[90.0], [np.nan]], atol=1e-5)

    @pytest.mark.parametrize("depth", [0.5, np.nan, np.inf])
    def test_third_coordinate_is_ignored(self, depth: float) -> None:
        """Angles use x/y only, so depth never changes or invalidates them."""
        xy = np.array([[[90, 50, depth], [50, 50, depth], [50, 10, depth]]])
        key_points = sv.KeyPoints(xy=xy.astype(np.float32))

        angles = sv.joint_angles(key_points, [JOINT])

        np.testing.assert_allclose(angles, [[90.0]], atol=1e-5)

    @pytest.mark.parametrize("num_points", [0, 17])
    def test_zero_skeletons_give_zero_rows(self, num_points: int) -> None:
        """A frame without skeletons returns zero rows, one column per joint."""
        key_points = sv.KeyPoints(xy=np.zeros((0, num_points, 2), dtype=np.float32))

        angles = sv.joint_angles(key_points, [JOINT, JOINT])

        assert angles.shape == (0, 2)

    def test_no_joints_give_empty_columns(self) -> None:
        """Without joints, each skeleton still gets a row."""
        key_points = _single_joint([90.0, 50.0], [50.0, 10.0])

        angles = sv.joint_angles(key_points, [])

        assert angles.shape == (1, 0)

    @pytest.mark.parametrize(
        ("num_points", "joint"),
        [
            pytest.param(3, JointAngle(name="too_far", a=0, b=1, c=3), id="past-end"),
            pytest.param(0, JointAngle(name="too_far", a=0, b=1, c=2), id="no-points"),
        ],
    )
    def test_index_out_of_range_raises(
        self, num_points: int, joint: JointAngle
    ) -> None:
        """A joint that points past the last key point raises ValueError."""
        key_points = sv.KeyPoints(xy=np.ones((1, num_points, 2), dtype=np.float32))

        with pytest.raises(ValueError, match="'too_far' refers to position"):
            sv.joint_angles(key_points, [joint])

    def test_input_is_not_modified(self) -> None:
        """Computing angles leaves the key points untouched."""
        key_points = _single_joint([np.nan, 50.0], [50.0, 10.0])
        xy_before = key_points.xy.copy()

        sv.joint_angles(key_points, [JOINT])

        np.testing.assert_array_equal(key_points.xy, xy_before)
