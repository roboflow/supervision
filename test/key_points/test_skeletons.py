from supervision.key_points.skeletons import (
    SKELETONS_BY_EDGE_COUNT,
    SKELETONS_BY_VERTEX_COUNT,
    Skeleton,
)


class TestSkeletons:
    def test_skeleton_enum_values(self):
        """Test skeleton enum has correct structure."""
        for skeleton in Skeleton:
            assert isinstance(skeleton.value, tuple)
            assert all(
                isinstance(edge, tuple) and len(edge) == 2 for edge in skeleton.value
            )

    def test_skeletons_by_vertex_count(self):
        """Test SKELETONS_BY_VERTEX_COUNT dictionary population."""
        # Test that the dictionary is populated
        assert len(SKELETONS_BY_VERTEX_COUNT) > 0

        # Test specific known skeletons
        coco_skeleton = Skeleton.COCO.value
        assert 17 in SKELETONS_BY_VERTEX_COUNT  # COCO has 17 keypoints
        assert SKELETONS_BY_VERTEX_COUNT[17] == coco_skeleton

    def test_skeletons_by_edge_count(self):
        """Test SKELETONS_BY_EDGE_COUNT dictionary mapping."""
        # Test that the dictionary is populated
        assert len(SKELETONS_BY_EDGE_COUNT) > 0

        # Test that edges are correctly counted
        for skeleton in Skeleton:
            edge_count = len(skeleton.value)
            assert edge_count in SKELETONS_BY_EDGE_COUNT
            assert SKELETONS_BY_EDGE_COUNT[edge_count] == skeleton.value

    def test_unique_vertices_calculation(self):
        """Test unique vertices calculation from skeleton edges."""
        coco_skeleton = Skeleton.COCO.value
        unique_vertices = {vertex for edge in coco_skeleton for vertex in edge}
        assert len(unique_vertices) == 17  # COCO has 17 keypoints

    def test_no_duplicate_skeletons_by_vertex_count(self):
        """Test no duplicate vertex counts across skeletons."""
        vertex_counts = [len({v for edge in s.value for v in edge}) for s in Skeleton]
        assert len(vertex_counts) == len(set(vertex_counts))
