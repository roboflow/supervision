"""
Unit tests for COCO dataset splitting functionality.
"""

import json
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

# Import the functions to test
from supervision.dataset.utils import split_coco_dataset, verify_coco_split, split_data


class TestCOCODatasetSplitting:
    """Test cases for COCO dataset splitting functionality."""
    
    @pytest.fixture
    def sample_coco_data(self):
        """Create comprehensive sample COCO data for testing."""
        return {
            "info": {
                "description": "Test COCO Dataset",
                "version": "1.0",
                "year": 2023
            },
            "licenses": [
                {"id": 1, "name": "Test License", "url": "http://test.com"}
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "person"},
                {"id": 2, "name": "car", "supercategory": "vehicle"},
                {"id": 3, "name": "bicycle", "supercategory": "vehicle"}
            ],
            "images": [
                {"id": 1, "file_name": "image001.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "image002.jpg", "width": 640, "height": 480},
                {"id": 3, "file_name": "image003.jpg", "width": 640, "height": 480},
                {"id": 4, "file_name": "image004.jpg", "width": 640, "height": 480},
                {"id": 5, "file_name": "image005.jpg", "width": 640, "height": 480},
                {"id": 6, "file_name": "image006.jpg", "width": 640, "height": 480},
                {"id": 7, "file_name": "image007.jpg", "width": 640, "height": 480},
                {"id": 8, "file_name": "image008.jpg", "width": 640, "height": 480},
                {"id": 9, "file_name": "image009.jpg", "width": 640, "height": 480},
                {"id": 10, "file_name": "image010.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100, 100, 60, 60], "area": 3600, "iscrowd": 0},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [20, 20, 60, 60], "area": 3600, "iscrowd": 0},
                {"id": 4, "image_id": 3, "category_id": 2, "bbox": [30, 30, 70, 70], "area": 4900, "iscrowd": 0},
                {"id": 5, "image_id": 4, "category_id": 3, "bbox": [40, 40, 80, 80], "area": 6400, "iscrowd": 0},
                {"id": 6, "image_id": 5, "category_id": 1, "bbox": [50, 50, 90, 90], "area": 8100, "iscrowd": 0},
                {"id": 7, "image_id": 6, "category_id": 2, "bbox": [60, 60, 100, 100], "area": 10000, "iscrowd": 0},
                {"id": 8, "image_id": 7, "category_id": 3, "bbox": [70, 70, 110, 110], "area": 12100, "iscrowd": 0},
                {"id": 9, "image_id": 8, "category_id": 1, "bbox": [80, 80, 120, 120], "area": 14400, "iscrowd": 0},
                {"id": 10, "image_id": 9, "category_id": 2, "bbox": [90, 90, 130, 130], "area": 16900, "iscrowd": 0},
                {"id": 11, "image_id": 10, "category_id": 3, "bbox": [100, 100, 140, 140], "area": 19600, "iscrowd": 0}
            ]
        }
    
    @pytest.fixture
    def setup_test_dataset(self, sample_coco_data):
        """Create temporary dataset files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create annotations file
            annotations_file = temp_path / "annotations.json"
            with open(annotations_file, 'w') as f:
                json.dump(sample_coco_data, f)
            
            # Create images directory and dummy image files
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            for i in range(1, 11):
                img_file = images_dir / f"image{i:03d}.jpg"
                # Create a simple test image
                img = Image.new('RGB', (640, 480), color='red')
                img.save(img_file)
            
            yield {
                'annotations_file': annotations_file,
                'images_dir': images_dir,
                'temp_dir': temp_path
            }
    
    def test_split_coco_dataset_basic(self, setup_test_dataset):
        """Test basic dataset splitting functionality."""
        data = setup_test_dataset
        output_dir = data['temp_dir'] / "splits"
        
        # Set random seed for reproducible tests
        stats = split_coco_dataset(
            annotations_path=data['annotations_file'],
            images_directory=data['images_dir'],
            output_directory=output_dir,
            val_percentage=0.2,
            test_percentage=0.1,
            random_state=42,
            verify_images=True
        )
        
        # Check statistics structure
        assert 'train' in stats
        assert 'val' in stats
        assert 'test' in stats
        assert 'total' in stats
        
        # Check that splits sum to total
        total_images = stats['train']['images'] + stats['val']['images'] + stats['test']['images']
        assert total_images == stats['total']['images']
        
        # Check that files were created
        assert (output_dir / "train_annotations.json").exists()
        assert (output_dir / "val_annotations.json").exists()
        assert (output_dir / "test_annotations.json").exists()
        
        # Verify split proportions are approximately correct
        total = stats['total']['images']
        assert abs(stats['val']['images'] / total - 0.2) < 0.1  # Allow some variance due to rounding
        assert abs(stats['test']['images'] / total - 0.1) < 0.1
    
    def test_split_coco_dataset_without_verification(self, setup_test_dataset):
        """Test dataset splitting without image verification."""
        data = setup_test_dataset
        output_dir = data['temp_dir'] / "splits_no_verify"
        
        stats = split_coco_dataset(
            annotations_path=data['annotations_file'],
            images_directory=data['images_dir'],
            output_directory=output_dir,
            val_percentage=0.3,
            test_percentage=0.2,
            random_state=42,
            verify_images=False
        )
        
        # Should process all images since verification is disabled
        assert stats['total']['images'] == 10
        assert stats['total']['annotations'] == 11
    
    def test_split_coco_dataset_invalid_inputs(self, setup_test_dataset):
        """Test error handling for invalid inputs."""
        data = setup_test_dataset
        
        # Test invalid percentages
        with pytest.raises(ValueError, match="Percentages must be between 0 and 1"):
            split_coco_dataset(
                annotations_path=data['annotations_file'],
                images_directory=data['images_dir'],
                output_directory=data['temp_dir'] / "test",
                val_percentage=1.5,  # Invalid
                test_percentage=0.1
            )
        
        # Test percentages that sum >= 1.0
        with pytest.raises(ValueError, match="Sum of val_percentage and test_percentage must be < 1.0"):
            split_coco_dataset(
                annotations_path=data['annotations_file'],
                images_directory=data['images_dir'],
                output_directory=data['temp_dir'] / "test",
                val_percentage=0.7,
                test_percentage=0.5  # Sum = 1.2
            )
        
        # Test non-existent annotations file
        with pytest.raises(FileNotFoundError):
            split_coco_dataset(
                annotations_path="nonexistent.json",
                images_directory=data['images_dir'],
                output_directory=data['temp_dir'] / "test"
            )
        
        # Test non-existent images directory
        with pytest.raises(FileNotFoundError):
            split_coco_dataset(
                annotations_path=data['annotations_file'],
                images_directory="nonexistent_dir",
                output_directory=data['temp_dir'] / "test"
            )
    
    def test_split_coco_dataset_min_annotations(self, setup_test_dataset):
        """Test filtering by minimum annotations per image."""
        data = setup_test_dataset
        output_dir = data['temp_dir'] / "splits_min_ann"
        
        stats = split_coco_dataset(
            annotations_path=data['annotations_file'],
            images_directory=data['images_dir'],
            output_directory=output_dir,
            val_percentage=0.2,
            test_percentage=0.1,
            random_state=42,
            min_annotations_per_image=2,  # Only image 1 has 2+ annotations
            verify_images=False
        )
        
        # Should only include image 1 (which has 2 annotations)
        assert stats['total']['images'] == 1
        assert stats['total']['annotations'] == 2
    
    def test_split_coco_dataset_reproducibility(self, setup_test_dataset):
        """Test that splits are reproducible with same random seed."""
        data = setup_test_dataset
        
        # First split
        stats1 = split_coco_dataset(
            annotations_path=data['annotations_file'],
            images_directory=data['images_dir'],
            output_directory=data['temp_dir'] / "splits1",
            val_percentage=0.2,
            test_percentage=0.1,
            random_state=42,
            verify_images=False
        )
        
        # Second split with same seed
        stats2 = split_coco_dataset(
            annotations_path=data['annotations_file'],
            images_directory=data['images_dir'],
            output_directory=data['temp_dir'] / "splits2",
            val_percentage=0.2,
            test_percentage=0.1,
            random_state=42,
            verify_images=False
        )
        
        # Results should be identical
        assert stats1 == stats2
    
    def test_verify_coco_split(self, setup_test_dataset):
        """Test COCO split verification functionality."""
        data = setup_test_dataset
        output_dir = data['temp_dir'] / "splits_verify"
        
        # Create splits
        split_coco_dataset(
            annotations_path=data['annotations_file'],
            images_directory=data['images_dir'],
            output_directory=output_dir,
            val_percentage=0.2,
            test_percentage=0.1,
            random_state=42,
            verify_images=False
        )
        
        # Verify splits
        verification = verify_coco_split(
            train_annotations=output_dir / "train_annotations.json",
            val_annotations=output_dir / "val_annotations.json",
            test_annotations=output_dir / "test_annotations.json",
            original_annotations=data['annotations_file']
        )
        
        # All verifications should pass
        assert verification['valid_format'] is True
        assert verification['no_duplicates'] is True
        assert verification['complete_split'] is True
        assert verification['categories_preserved'] is True
        assert 'details' in verification
    
    def test_verify_coco_split_with_duplicates(self, setup_test_dataset):
        """Test split verification detects duplicates."""
        data = setup_test_dataset
        output_dir = data['temp_dir'] / "splits_duplicate"
        output_dir.mkdir()
        
        # Create a split with duplicated images (manually)
        with open(data['annotations_file']) as f:
            original_data = json.load(f)
        
        # Create train split with first 5 images
        train_data = {
            'info': original_data['info'],
            'licenses': original_data['licenses'],
            'categories': original_data['categories'],
            'images': original_data['images'][:5],
            'annotations': [ann for ann in original_data['annotations'] if ann['image_id'] <= 5]
        }
        
        # Create val split with overlapping images (1-3)
        val_data = {
            'info': original_data['info'],
            'licenses': original_data['licenses'],
            'categories': original_data['categories'],
            'images': original_data['images'][:3],  # Duplicate first 3 images
            'annotations': [ann for ann in original_data['annotations'] if ann['image_id'] <= 3]
        }
        
        # Create test split with remaining images
        test_data = {
            'info': original_data['info'],
            'licenses': original_data['licenses'],
            'categories': original_data['categories'],
            'images': original_data['images'][5:],
            'annotations': [ann for ann in original_data['annotations'] if ann['image_id'] > 5]
        }
        
        # Save the splits
        with open(output_dir / "train_annotations.json", 'w') as f:
            json.dump(train_data, f)
        with open(output_dir / "val_annotations.json", 'w') as f:
            json.dump(val_data, f)
        with open(output_dir / "test_annotations.json", 'w') as f:
            json.dump(test_data, f)
        
        # Verify - should detect duplicates
        verification = verify_coco_split(
            train_annotations=output_dir / "train_annotations.json",
            val_annotations=output_dir / "val_annotations.json",
            test_annotations=output_dir / "test_annotations.json",
            original_annotations=data['annotations_file']
        )
        
        assert verification['no_duplicates'] is False
        assert 'duplicate_images' in verification['details']
    
    def test_split_data_backward_compatibility(self, setup_test_dataset):
        """Test backward compatibility with original split_data function."""
        data = setup_test_dataset
        
        # Test with warning capture
        with pytest.warns(DeprecationWarning, match="split_data\\(\\) is deprecated"):
            stats = split_data(
                root=data['annotations_file'],
                imgs_path=data['images_dir'],
                val_perc=0.2,
                test_perc=0.1
            )
        
        # Should return valid statistics
        assert 'train' in stats
        assert 'val' in stats
        assert 'test' in stats
        assert 'total' in stats
    
    def test_coco_format_preservation(self, setup_test_dataset):
        """Test that output files maintain proper COCO format."""
        data = setup_test_dataset
        output_dir = data['temp_dir'] / "splits_format"
        
        split_coco_dataset(
            annotations_path=data['annotations_file'],
            images_directory=data['images_dir'],
            output_directory=output_dir,
            val_percentage=0.3,
            test_percentage=0.2,
            random_state=42,
            verify_images=False
        )
        
        # Check that each split file has proper COCO structure
        for split_name in ['train', 'val', 'test']:
            split_file = output_dir / f"{split_name}_annotations.json"
            if split_file.exists():
                with open(split_file) as f:
                    split_data = json.load(f)
                
                # Check required COCO fields
                required_fields = ['info', 'licenses', 'categories', 'images', 'annotations']
                for field in required_fields:
                    assert field in split_data, f"Missing {field} in {split_name} split"
                
                # Check that categories are complete
                assert len(split_data['categories']) == 3  # All categories should be preserved
                
                # Check that image IDs in annotations match images
                image_ids = set(img['id'] for img in split_data['images'])
                annotation_image_ids = set(ann['image_id'] for ann in split_data['annotations'])
                assert annotation_image_ids.issubset(image_ids)
    
    def test_empty_dataset_handling(self):
        """Test handling of empty or minimal datasets."""
        # Create minimal COCO dataset with no images
        empty_data = {
            "info": {"description": "Empty dataset"},
            "licenses": [],
            "categories": [{"id": 1, "name": "test", "supercategory": "test"}],
            "images": [],
            "annotations": []
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create annotations file
            annotations_file = temp_path / "empty_annotations.json"
            with open(annotations_file, 'w') as f:
                json.dump(empty_data, f)
            
            # Create empty images directory
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Should handle empty dataset gracefully
            stats = split_coco_dataset(
                annotations_path=annotations_file,
                images_directory=images_dir,
                output_directory=temp_path / "splits",
                val_percentage=0.2,
                test_percentage=0.1,
                verify_images=False
            )
            
            # All splits should be empty
            assert stats['total']['images'] == 0
            assert stats['total']['annotations'] == 0
    
    def test_missing_images_handling(self, setup_test_dataset):
        """Test handling when some images are missing from filesystem."""
        data = setup_test_dataset
        
        # Remove some image files
        (data['images_dir'] / "image005.jpg").unlink()
        (data['images_dir'] / "image008.jpg").unlink()
        
        output_dir = data['temp_dir'] / "splits_missing"
        
        # Should skip missing images with warning
        stats = split_coco_dataset(
            annotations_path=data['annotations_file'],
            images_directory=data['images_dir'],
            output_directory=output_dir,
            val_percentage=0.2,
            test_percentage=0.1,
            random_state=42,
            verify_images=True
        )
        
        # Should have fewer than 10 images due to missing files
        assert stats['total']['images'] < 10
    
    @patch('supervision.dataset.utils.COCO')
    def test_coco_loading_error_handling(self, mock_coco, setup_test_dataset):
        """Test error handling when COCO loading fails."""
        data = setup_test_dataset
        
        # Mock COCO to raise exception
        mock_coco.side_effect = Exception("Failed to load COCO dataset")
        
        with pytest.raises(Exception, match="Failed to load COCO dataset"):
            split_coco_dataset(
                annotations_path=data['annotations_file'],
                images_directory=data['images_dir'],
                output_directory=data['temp_dir'] / "splits"
            )


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""
    
    def test_large_dataset_simulation(self):
        """Test with simulated large dataset structure."""
        # Create a larger simulated dataset
        large_data = {
            "info": {"description": "Large test dataset"},
            "licenses": [],
            "categories": [
                {"id": i, "name": f"class_{i}", "supercategory": "object"} 
                for i in range(1, 21)  # 20 categories
            ],
            "images": [
                {"id": i, "file_name": f"image_{i:06d}.jpg", "width": 1920, "height": 1080}
                for i in range(1, 1001)  # 1000 images
            ],
            "annotations": []
        }
        
        # Generate random annotations
        np.random.seed(42)
        ann_id = 1
        for img_id in range(1, 1001):
            # Random number of annotations per image (1-5)
            num_annotations = np.random.randint(1, 6)
            for _ in range(num_annotations):
                large_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": np.random.randint(1, 21),
                    "bbox": [
                        np.random.randint(0, 1800),
                        np.random.randint(0, 980),
                        np.random.randint(50, 200),
                        np.random.randint(50, 200)
                    ],
                    "area": np.random.randint(2500, 40000),
                    "iscrowd": 0
                })
                ann_id += 1
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create annotations file
            annotations_file = temp_path / "large_annotations.json"
            with open(annotations_file, 'w') as f:
                json.dump(large_data, f)
            
            # Create images directory (but don't create actual images for speed)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Test splitting without image verification for speed
            stats = split_coco_dataset(
                annotations_path=annotations_file,
                images_directory=images_dir,
                output_directory=temp_path / "splits",
                val_percentage=0.15,
                test_percentage=0.15,
                random_state=42,
                verify_images=False  # Skip verification for speed
            )
            
            # Check proportions are correct
            total_images = stats['total']['images']
            assert total_images == 1000
            
            # Check split proportions (allowing for rounding)
            val_ratio = stats['val']['images'] / total_images
            test_ratio = stats['test']['images'] / total_images
            train_ratio = stats['train']['images'] / total_images
            
            assert abs(val_ratio - 0.15) < 0.02  # Within 2%
            assert abs(test_ratio - 0.15) < 0.02
            assert abs(train_ratio - 0.70) < 0.02
    
    def test_real_world_edge_cases(self):
        """Test edge cases that might occur in real datasets."""
        # Dataset with some images having no annotations
        edge_case_data = {
            "info": {"description": "Edge case dataset"},
            "licenses": [],
            "categories": [{"id": 1, "name": "object", "supercategory": "thing"}],
            "images": [
                {"id": 1, "file_name": "image_with_ann.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "image_no_ann.jpg", "width": 640, "height": 480},
                {"id": 3, "file_name": "image_with_ann2.jpg", "width": 640, "height": 480},
                {"id": 4, "file_name": "image_no_ann2.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
                {"id": 2, "image_id": 3, "category_id": 1, "bbox": [20, 20, 60, 60], "area": 3600, "iscrowd": 0}
                # Images 2 and 4 have no annotations
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            annotations_file = temp_path / "edge_case_annotations.json"
            with open(annotations_file, 'w') as f:
                json.dump(edge_case_data, f)
            
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Test with min_annotations_per_image = 1 (default)
            stats = split_coco_dataset(
                annotations_path=annotations_file,
                images_directory=images_dir,
                output_directory=temp_path / "splits",
                val_percentage=0.25,
                test_percentage=0.25,
                verify_images=False,
                min_annotations_per_image=1
            )
            
            # Should only include 2 images (those with annotations)
            assert stats['total']['images'] == 2
            assert stats['total']['annotations'] == 2


if __name__ == "__main__":
    pytest.main([__file__])