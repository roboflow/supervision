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

        # Reconstruct the expected mapping: for each skeleton, map the number of
        # edges in skeleton.value to skeleton.value itself (as done in skeletons.py).
        expected = {}
        for skeleton in Skeleton:
            edge_count = len(skeleton.value)
            expected[edge_count] = skeleton.value

        assert SKELETONS_BY_EDGE_COUNT == expected

    def test_unique_vertices_calculation(self):
        """Test unique vertices calculation from skeleton edges."""
        coco_skeleton = Skeleton.COCO.value
        unique_vertices = {vertex for edge in coco_skeleton for vertex in edge}
        assert len(unique_vertices) == 17  # COCO has 17 keypoints

    def test_skeletons_by_vertex_count_mapping_behaviour(self):
        """Test SKELETONS_BY_VERTEX_COUNT uses last-in-wins for duplicate counts."""
        expected_mapping = {}
        for skeleton in Skeleton:
            vertex_count = len({v for edge in skeleton.value for v in edge})
            # Mimic skeletons.py: later skeletons overwrite earlier ones
            expected_mapping[vertex_count] = skeleton.value

        # The keys (vertex counts) should match
        assert set(SKELETONS_BY_VERTEX_COUNT.keys()) == set(expected_mapping.keys())

        # For each vertex count, the stored skeleton should be the last one encountered
        for vertex_count, skeleton_value in expected_mapping.items():
            assert SKELETONS_BY_VERTEX_COUNT[vertex_count] == skeleton_value
