"""Tests for Copernicus S2 image processing, indices, quality control, and visualization."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.data.copernicus.image_processing import (
    crop_to_bbox,
    get_image_statistics,
)
from src.data.copernicus.indices import (
    calculate_evi,
    calculate_nbr,
    calculate_ndvi,
    calculate_ndwi,
    calculate_savi,
)
from src.data.copernicus.quality import apply_cloud_mask_to_image


class TestImageProcessing:
    """Test image processing functions."""

    def test_crop_to_bbox_valid(self):
        """Test cropping with valid bbox."""
        # Create test image (100x100 RGB)
        image = np.random.rand(100, 100, 3).astype(np.float32)
        image_bounds = [0.0, 0.0, 1.0, 1.0]  # Full image bounds
        target_bbox = [0.25, 0.25, 0.75, 0.75]  # Center quarter

        cropped = crop_to_bbox(image, image_bounds, target_bbox)

        assert cropped is not None
        assert cropped.shape[2] == 3  # RGB channels preserved
        assert cropped.shape[0] < image.shape[0]  # Height reduced
        assert cropped.shape[1] < image.shape[1]  # Width reduced

    def test_crop_to_bbox_outside_bounds(self):
        """Test cropping with bbox outside image bounds."""
        image = np.random.rand(100, 100, 3).astype(np.float32)
        image_bounds = [0.0, 0.0, 1.0, 1.0]
        target_bbox = [2.0, 2.0, 3.0, 3.0]  # Completely outside

        cropped = crop_to_bbox(image, image_bounds, target_bbox)

        assert cropped is None

    def test_crop_to_bbox_2d_image(self):
        """Test cropping with 2D (grayscale) image."""
        image = np.random.rand(100, 100).astype(np.float32)
        image_bounds = [0.0, 0.0, 1.0, 1.0]
        target_bbox = [0.25, 0.25, 0.75, 0.75]

        cropped = crop_to_bbox(image, image_bounds, target_bbox)

        assert cropped is not None
        assert cropped.ndim == 2  # Still 2D
        assert cropped.shape[0] < image.shape[0]

    def test_get_image_statistics(self):
        """Test image statistics calculation."""
        # Create mock RGB data
        rgb_array = np.random.rand(100, 100, 3).astype(np.float32)
        rgb_data = {
            "rgb_array": rgb_array,
            "bounds_wgs84": (6.0, 49.0, 7.0, 50.0),
        }

        stats = get_image_statistics(rgb_data)

        assert "shape" in stats
        assert "min_values" in stats
        assert "max_values" in stats
        assert "mean_values" in stats
        assert "std_values" in stats
        assert "coverage_area_km2" in stats
        assert stats["shape"] == (100, 100, 3)
        assert len(stats["min_values"]) == 3  # RGB channels

    def test_get_image_statistics_none_input(self):
        """Test statistics with None input."""
        stats = get_image_statistics(None)
        assert stats == {}


class TestSpectralIndices:
    """Test spectral index calculations."""

    @patch("src.data.copernicus.indices._extract_band")
    def test_calculate_ndvi(self, mock_extract):
        """Test NDVI calculation."""
        # Mock NIR and Red bands
        nir = np.ones((100, 100), dtype=np.float32) * 0.8
        red = np.ones((100, 100), dtype=np.float32) * 0.2

        mock_extract.side_effect = [nir, red]

        result = calculate_ndvi(Path("fake.zip"))

        assert result is not None
        assert "ndvi" in result
        assert "metadata" in result
        assert result["ndvi"].shape == (100, 100)
        # NDVI = (0.8 - 0.2) / (0.8 + 0.2) = 0.6
        assert np.allclose(result["ndvi"], 0.6, atol=0.01)

    @patch("src.data.copernicus.indices._extract_band")
    def test_calculate_ndvi_missing_band(self, mock_extract):
        """Test NDVI with missing band."""
        mock_extract.side_effect = [None, None]

        result = calculate_ndvi(Path("fake.zip"))

        assert result is None

    @patch("src.data.copernicus.indices._extract_band")
    def test_calculate_ndwi(self, mock_extract):
        """Test NDWI calculation."""
        green = np.ones((100, 100), dtype=np.float32) * 0.6
        nir = np.ones((100, 100), dtype=np.float32) * 0.2

        mock_extract.side_effect = [green, nir]

        result = calculate_ndwi(Path("fake.zip"))

        assert result is not None
        assert "ndwi" in result
        # NDWI = (0.6 - 0.2) / (0.6 + 0.2) = 0.5
        assert np.allclose(result["ndwi"], 0.5, atol=0.01)

    @patch("src.data.copernicus.indices._extract_band")
    def test_calculate_evi(self, mock_extract):
        """Test EVI calculation."""
        nir = np.ones((100, 100), dtype=np.float32) * 0.8
        red = np.ones((100, 100), dtype=np.float32) * 0.2
        blue = np.ones((100, 100), dtype=np.float32) * 0.1

        mock_extract.side_effect = [nir, red, blue]

        result = calculate_evi(Path("fake.zip"))

        assert result is not None
        assert "evi" in result
        assert result["evi"].shape == (100, 100)

    @patch("src.data.copernicus.indices._extract_band")
    def test_calculate_savi(self, mock_extract):
        """Test SAVI calculation."""
        nir = np.ones((100, 100), dtype=np.float32) * 0.8
        red = np.ones((100, 100), dtype=np.float32) * 0.2

        mock_extract.side_effect = [nir, red]

        result = calculate_savi(Path("fake.zip"), L=0.5)

        assert result is not None
        assert "savi" in result
        assert result["metadata"]["L_factor"] == 0.5

    @patch("src.data.copernicus.indices._extract_band")
    def test_calculate_nbr(self, mock_extract):
        """Test NBR calculation."""
        nir = np.ones((100, 100), dtype=np.float32) * 0.8
        swir = np.ones((100, 100), dtype=np.float32) * 0.2

        mock_extract.side_effect = [nir, swir]

        result = calculate_nbr(Path("fake.zip"))

        assert result is not None
        assert "nbr" in result
        # NBR = (0.8 - 0.2) / (0.8 + 0.2) = 0.6
        assert np.allclose(result["nbr"], 0.6, atol=0.01)


class TestQualityControl:
    """Test quality control functions."""

    def test_apply_cloud_mask_2d(self):
        """Test applying cloud mask to 2D image."""
        image = np.ones((100, 100), dtype=np.float32)
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 0  # Cloud in center

        masked = apply_cloud_mask_to_image(image, mask, fill_value=0.0)

        assert masked.shape == image.shape
        assert masked[50, 50] == 0.0  # Cloudy pixel masked
        assert masked[10, 10] == 1.0  # Clear pixel preserved

    def test_apply_cloud_mask_3d(self):
        """Test applying cloud mask to 3D RGB image."""
        image = np.ones((100, 100, 3), dtype=np.float32)
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 0  # Cloud in center

        masked = apply_cloud_mask_to_image(image, mask, fill_value=0.0)

        assert masked.shape == image.shape
        assert np.all(masked[50, 50, :] == 0.0)  # All channels masked
        assert np.all(masked[10, 10, :] == 1.0)  # All channels preserved

    def test_apply_cloud_mask_nan_fill(self):
        """Test applying cloud mask with NaN fill value."""
        image = np.ones((100, 100), dtype=np.float32)
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 0

        masked = apply_cloud_mask_to_image(image, mask, fill_value=np.nan)

        assert np.isnan(masked[50, 50])
        assert masked[10, 10] == 1.0

    def test_apply_cloud_mask_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        image = np.ones((100, 100, 3, 3), dtype=np.float32)  # 4D invalid
        mask = np.ones((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError):
            apply_cloud_mask_to_image(image, mask)


class TestVisualization:
    """Test visualization functions (mock matplotlib)."""

    @patch("src.data.copernicus.visualization.extract_rgb_composite")
    @patch("matplotlib.pyplot.subplots")
    def test_display_satellite_image(self, mock_subplots, mock_extract):
        """Test satellite image display."""
        from src.data.copernicus.visualization import display_satellite_image

        # Mock RGB data
        mock_extract.return_value = {
            "rgb_array": np.random.rand(100, 100, 3),
            "bounds_wgs84": (6.0, 49.0, 7.0, 50.0),
        }

        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = display_satellite_image(
            Path("fake.zip"), target_bbox=[6.1, 49.1, 6.2, 49.2], ax=mock_ax
        )

        assert result is not None
        mock_ax.imshow.assert_called_once()
        mock_ax.plot.assert_called()  # Target bbox overlay

    @patch("src.data.copernicus.visualization.extract_rgb_composite")
    def test_display_satellite_image_extraction_failed(self, mock_extract):
        """Test handling of extraction failure."""
        from src.data.copernicus.visualization import display_satellite_image

        mock_extract.return_value = None

        result = display_satellite_image(Path("fake.zip"), target_bbox=[6.1, 49.1, 6.2, 49.2])

        assert result is None

    @patch("matplotlib.pyplot.subplots")
    def test_create_coverage_map(self, mock_subplots):
        """Test coverage map creation."""
        from src.data.copernicus.visualization import create_coverage_map

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = create_coverage_map(
            target_bbox=[6.0, 49.0, 7.0, 50.0],
            center_lat=49.5,
            center_lon=6.5,
            s2_files=[Path("file1.zip"), Path("file2.zip")],
            ax=mock_ax,
        )

        assert result is not None
        mock_ax.plot.assert_called()  # Should plot bbox and coverage

    @patch("matplotlib.pyplot.subplots")
    def test_create_metadata_summary_no_files(self, mock_subplots):
        """Test metadata summary with no files."""
        from src.data.copernicus.visualization import create_metadata_summary

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = create_metadata_summary([], ax=mock_ax)

        assert result is not None
        mock_ax.text.assert_called()  # Should show "no data" message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
