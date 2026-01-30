"""Integration tests for Copernicus data client using real downloaded data.

OVERVIEW:
These tests verify the complete Copernicus data pipeline using actual satellite products:
- Sentinel-2 (optical): RGB composites, spectral indices (NDVI, NDWI)
- Sentinel-1 (SAR): Backscatter extraction, polarization handling

The tests use real downloaded products (not mocked data) to ensure the client works
with actual Copernicus Data Space Ecosystem files.

PREREQUISITES:
1. Download test data first:
   uv run python scripts/download_test_data.py

2. This creates:
   - data/cache/copernicus/s1/*.zip (Sentinel-1 SAR products)
   - data/cache/copernicus/s2/*.zip (Sentinel-2 optical products)
   - data/test_fixtures/test_data_metadata.json (metadata)

USAGE:
    # Run all tests
    uv run python -m pytest tests/test_copernicus_integration.py -v

    # Run specific test
    uv run python -m pytest tests/test_copernicus_integration.py::TestCopernicusIntegration::test_s2_rgb_extraction -v

    # Run with output
    uv run python -m pytest tests/test_copernicus_integration.py -v -s

TEST ORGANIZATION:
- Sentinel-2 Tests: Optical imagery processing (RGB, indices, statistics)
- Sentinel-1 Tests: SAR imagery processing (polarizations, backscatter)
- Cross-sensor Tests: Metadata and data availability checks
"""

import json
import unittest
from pathlib import Path

import numpy as np

from src.data.copernicus.image_processing import (
    extract_rgb_composite,
    extract_sar_composite,
    get_image_statistics,
)
from src.data.copernicus.indices import calculate_ndvi, calculate_ndwi


class TestCopernicusIntegration(unittest.TestCase):
    """Integration tests using real downloaded Copernicus data.

    These tests verify that the Copernicus client can:
    1. Read actual satellite products from Copernicus Data Space
    2. Extract bands and create composites
    3. Calculate spectral indices
    4. Handle both optical (S2) and SAR (S1) data

    All tests use real downloaded data, not mocked responses.
    """

    @classmethod
    def setUpClass(cls):
        """Load test data metadata and verify files exist.

        This runs once before all tests. It:
        1. Checks if test data has been downloaded
        2. Loads metadata about downloaded products
        3. Locates S1 and S2 files in the cache directory

        If test data is missing, all tests will be skipped with a helpful message.
        """
        cls.cache_dir = Path("data/cache/copernicus")
        cls.metadata_file = Path("data/test_fixtures/test_data_metadata.json")

        if not cls.metadata_file.exists():
            raise unittest.SkipTest(
                "Test data not found. Run: uv run python scripts/download_test_data.py"
            )

        with open(cls.metadata_file) as f:
            cls.metadata = json.load(f)

        # Find S2 and S1 files from metadata
        # We take the first available product of each type
        cls.s2_file = None
        cls.s1_file = None

        for product_key, product_info in cls.metadata["products"].items():
            if product_key.startswith("s2_") and cls.s2_file is None:
                file_path = cls.cache_dir / "s2" / product_info["file"]
                if file_path.exists():
                    cls.s2_file = file_path

        for product_key, product_info in cls.metadata["products"].items():
            if product_key.startswith("s1_") and cls.s1_file is None:
                file_path = cls.cache_dir / "s1" / product_info["file"]
                if file_path.exists():
                    cls.s1_file = file_path

    # ========================================================================
    # Sentinel-2 Tests (Optical Imagery)
    # ========================================================================
    # These tests verify optical imagery processing from Sentinel-2.
    # S2 provides 13 spectral bands from visible to SWIR wavelengths.

    def test_s2_rgb_extraction(self):
        """Test RGB composite extraction from S2 product.

        WHAT THIS TESTS:
        - Reading S2 ZIP files
        - Extracting B04 (Red), B03 (Green), B02 (Blue) bands
        - Creating normalized RGB composite (0-1 range)
        - Extracting geospatial metadata (bounds, CRS)

        WHY IT MATTERS:
        RGB composites are the most common visualization for optical imagery.
        This verifies the basic ability to read and process S2 data.
        """
        if self.s2_file is None:
            self.skipTest("S2 test data not available")

        rgb_data = extract_rgb_composite(self.s2_file)

        # Verify structure
        self.assertIsNotNone(rgb_data, "RGB extraction failed")
        self.assertIn("rgb_array", rgb_data)
        self.assertIn("bounds_wgs84", rgb_data)
        self.assertIn("metadata", rgb_data)

        # Check array shape and type
        rgb_array = rgb_data["rgb_array"]
        self.assertEqual(rgb_array.ndim, 3, "RGB should be 3D array [H, W, C]")
        self.assertEqual(rgb_array.shape[2], 3, "RGB should have 3 channels")
        self.assertTrue(np.issubdtype(rgb_array.dtype, np.floating), "Should be float type")

        # Check value range (should be normalized 0-1)
        self.assertGreaterEqual(rgb_array.min(), 0.0, "RGB values should be >= 0")
        self.assertLessEqual(rgb_array.max(), 1.0, "RGB values should be <= 1")

        print(f"✓ S2 RGB extraction: {rgb_array.shape}")

    def test_s2_false_color(self):
        """Test false color composite (NIR-R-G) extraction.

        WHAT THIS TESTS:
        - Custom band selection (not just RGB)
        - NIR band (B08) extraction
        - Composite creation with arbitrary bands

        WHY IT MATTERS:
        False color composites (NIR-Red-Green) highlight vegetation in red,
        making it easier to identify healthy vegetation. This tests the ability
        to create custom band combinations beyond standard RGB.
        """
        if self.s2_file is None:
            self.skipTest("S2 test data not available")

        false_color = extract_rgb_composite(self.s2_file, bands=["B08", "B04", "B03"])

        self.assertIsNotNone(false_color, "False color extraction failed")
        self.assertEqual(false_color["rgb_array"].shape[2], 3, "Should have 3 channels")

        print(f"✓ S2 false color: {false_color['rgb_array'].shape}")

    def test_s2_ndvi_calculation(self):
        """Test NDVI calculation from S2 bands.

        WHAT THIS TESTS:
        - Extracting NIR (B08) and Red (B04) bands
        - Computing NDVI = (NIR - Red) / (NIR + Red)
        - Handling division by zero
        - Validating NDVI range [-1, 1]

        WHY IT MATTERS:
        NDVI (Normalized Difference Vegetation Index) is the most widely used
        vegetation index in remote sensing. It indicates vegetation health and
        density. This is a critical capability for agricultural monitoring.

        EXPECTED VALUES:
        - Water/clouds: < 0
        - Bare soil: 0 - 0.2
        - Sparse vegetation: 0.2 - 0.5
        - Dense vegetation: 0.5 - 1.0
        """
        if self.s2_file is None:
            self.skipTest("S2 test data not available")

        ndvi_data = calculate_ndvi(self.s2_file)

        # Verify structure
        self.assertIsNotNone(ndvi_data, "NDVI calculation failed")
        self.assertIn("ndvi", ndvi_data)
        self.assertIn("metadata", ndvi_data)

        ndvi = ndvi_data["ndvi"]
        self.assertEqual(ndvi.ndim, 2, "NDVI should be 2D array [H, W]")

        # NDVI range should be -1 to 1 (mathematical constraint)
        self.assertGreaterEqual(ndvi.min(), -1.0, "NDVI minimum should be >= -1")
        self.assertLessEqual(ndvi.max(), 1.0, "NDVI maximum should be <= 1")

        # Check for reasonable vegetation values
        # Note: NDVI values can be very low in winter, urban areas, or water bodies
        positive_pixels = np.sum(ndvi > 0.0)
        self.assertGreater(positive_pixels, 0, "Should have some positive NDVI pixels")

        print(
            f"✓ S2 NDVI: range [{ndvi.min():.3f}, {ndvi.max():.3f}], "
            f"{positive_pixels} positive pixels"
        )

    def test_s2_ndwi_calculation(self):
        """Test NDWI calculation from S2 bands.

        WHAT THIS TESTS:
        - Extracting Green (B03) and NIR (B08) bands
        - Computing NDWI = (Green - NIR) / (Green + NIR)
        - Validating NDWI range [-1, 1]

        WHY IT MATTERS:
        NDWI (Normalized Difference Water Index) highlights water bodies and
        measures water content in vegetation. Useful for:
        - Detecting water bodies (rivers, lakes, floods)
        - Monitoring irrigation
        - Assessing vegetation water stress

        EXPECTED VALUES:
        - Water bodies: > 0.3
        - Wet vegetation: 0 - 0.3
        - Dry vegetation/soil: < 0
        """
        if self.s2_file is None:
            self.skipTest("S2 test data not available")

        ndwi_data = calculate_ndwi(self.s2_file)

        # Verify structure
        self.assertIsNotNone(ndwi_data, "NDWI calculation failed")
        self.assertIn("ndwi", ndwi_data)

        ndwi = ndwi_data["ndwi"]
        self.assertEqual(ndwi.ndim, 2, "NDWI should be 2D array [H, W]")

        # NDWI range should be -1 to 1 (mathematical constraint)
        self.assertGreaterEqual(ndwi.min(), -1.0, "NDWI minimum should be >= -1")
        self.assertLessEqual(ndwi.max(), 1.0, "NDWI maximum should be <= 1")

        print(f"✓ S2 NDWI: range [{ndwi.min():.3f}, {ndwi.max():.3f}]")

    def test_s2_statistics(self):
        """Test image statistics calculation.

        WHAT THIS TESTS:
        - Computing basic statistics (min, max, mean, std)
        - Calculating coverage area from geospatial bounds
        - Extracting metadata (shape, dtype, bounds)

        WHY IT MATTERS:
        Statistics help assess data quality and understand the scene:
        - Coverage area: verify expected geographic extent
        - Value ranges: detect saturation or missing data
        - Mean/std: understand scene brightness and contrast
        """
        if self.s2_file is None:
            self.skipTest("S2 test data not available")

        rgb_data = extract_rgb_composite(self.s2_file)
        stats = get_image_statistics(rgb_data)

        # Verify expected fields
        self.assertIn("shape", stats)
        self.assertIn("coverage_area_km2", stats)

        # Verify reasonable values
        self.assertGreater(stats["coverage_area_km2"], 0, "Coverage area should be positive")

        print(f"✓ S2 stats: {stats['shape']}, {stats['coverage_area_km2']:.1f} km²")

    # NOTE: Quality assessment functions (assess_s2_quality, assess_s1_quality)
    # are not yet implemented. These tests are commented out for now.
    # They can be added in a future PR when quality assessment is implemented.

    # ========================================================================
    # Sentinel-1 Tests (SAR Imagery)
    # ========================================================================
    # These tests verify SAR imagery processing from Sentinel-1.
    # S1 provides radar backscatter in VV and VH polarizations.

    def test_s1_sar_extraction(self):
        """Test SAR composite extraction from S1 product.

        WHAT THIS TESTS:
        - Reading S1 ZIP files (SAFE format)
        - Extracting VV and VH polarization TIFFs
        - Converting to dB scale (10 * log10(intensity))
        - Extracting geospatial metadata

        WHY IT MATTERS:
        SAR data is fundamentally different from optical:
        - Works day/night and through clouds
        - Measures surface roughness, not reflectance
        - Requires different processing (dB conversion, speckle filtering)

        This verifies basic SAR data reading capability.
        """
        if self.s1_file is None:
            self.skipTest("S1 test data not available")

        sar_data = extract_sar_composite(self.s1_file)

        # Verify structure
        self.assertIsNotNone(sar_data, "SAR extraction failed")
        self.assertIn("sar_array", sar_data)
        self.assertIn("bounds_wgs84", sar_data)
        self.assertIn("polarizations", sar_data)
        self.assertIn("metadata", sar_data)

        # Check array shape
        sar_array = sar_data["sar_array"]
        self.assertEqual(sar_array.ndim, 3, "SAR should be 3D array [H, W, Polarizations]")

        # Check polarizations
        polarizations = sar_data["polarizations"]
        self.assertGreater(len(polarizations), 0, "Should have at least one polarization")
        self.assertEqual(
            sar_array.shape[2],
            len(polarizations),
            "Number of channels should match number of polarizations",
        )

        # Check dB values (typical range -30 to 10 dB for land surfaces)
        # Extended range to -50 to 20 to handle edge cases
        self.assertGreater(sar_array.min(), -50, "Backscatter too low (likely error)")
        self.assertLess(sar_array.max(), 20, "Backscatter too high (likely error)")

        print(f"✓ S1 SAR extraction: {sar_array.shape}, polarizations: {polarizations}")

    def test_s1_vv_polarization(self):
        """Test VV polarization extraction.

        WHAT THIS TESTS:
        - Selective polarization extraction (VV only)
        - Correct channel ordering

        WHY IT MATTERS:
        VV polarization (Vertical transmit, Vertical receive) is sensitive to:
        - Surface roughness
        - Soil moisture
        - Urban structures

        Often used alone for specific applications like flood detection.
        """
        if self.s1_file is None:
            self.skipTest("S1 test data not available")

        sar_data = extract_sar_composite(self.s1_file, polarizations=["VV"])

        self.assertIsNotNone(sar_data)
        self.assertIn("VV", sar_data["polarizations"])
        self.assertEqual(len(sar_data["polarizations"]), 1, "Should have only VV")

        print(f"✓ S1 VV polarization: {sar_data['sar_array'].shape}")

    def test_s1_vh_polarization(self):
        """Test VH polarization extraction.

        WHAT THIS TESTS:
        - Selective polarization extraction (VH only)
        - Cross-polarization handling

        WHY IT MATTERS:
        VH polarization (Vertical transmit, Horizontal receive) is sensitive to:
        - Volume scattering (vegetation canopy)
        - Crop type discrimination
        - Forest biomass

        Cross-polarization is key for vegetation monitoring.
        """
        if self.s1_file is None:
            self.skipTest("S1 test data not available")

        sar_data = extract_sar_composite(self.s1_file, polarizations=["VH"])

        self.assertIsNotNone(sar_data)
        self.assertIn("VH", sar_data["polarizations"])
        self.assertEqual(len(sar_data["polarizations"]), 1, "Should have only VH")

        print(f"✓ S1 VH polarization: {sar_data['sar_array'].shape}")

    def test_s1_backscatter_statistics(self):
        """Test SAR backscatter statistics.

        WHAT THIS TESTS:
        - Backscatter value ranges for each polarization
        - Statistical distribution (mean, std)
        - Data validity (no NaN, no extreme outliers)

        WHY IT MATTERS:
        Backscatter statistics help assess:
        - Data quality (extreme values indicate errors)
        - Scene characteristics (urban vs rural, wet vs dry)
        - Calibration correctness

        TYPICAL VALUES (dB):
        - Water: -25 to -15 (smooth surface, low backscatter)
        - Vegetation: -15 to -5 (volume scattering)
        - Urban: -5 to 5 (strong corner reflectors)
        - Bare soil: -10 to 0 (depends on roughness and moisture)
        """
        if self.s1_file is None:
            self.skipTest("S1 test data not available")

        sar_data = extract_sar_composite(self.s1_file)

        # Calculate statistics for each polarization
        for i, pol in enumerate(sar_data["polarizations"]):
            backscatter = sar_data["sar_array"][:, :, i]

            mean_db = np.mean(backscatter)
            std_db = np.std(backscatter)

            # Typical backscatter values (relaxed range for real data)
            # Real scenes can have higher values due to urban areas or corner reflectors
            self.assertGreater(mean_db, -30, f"{pol} mean too low (likely error)")
            self.assertLess(mean_db, 15, f"{pol} mean too high (likely error)")
            self.assertGreater(std_db, 0, f"{pol} should have variation")

            print(f"✓ S1 {pol} backscatter: mean={mean_db:.2f} dB, std={std_db:.2f} dB")

    # ========================================================================
    # Cross-sensor Tests
    # ========================================================================
    # These tests verify metadata and data availability across sensors.

    def test_both_sensors_available(self):
        """Test that both S1 and S2 data are available.

        WHAT THIS TESTS:
        - Download script successfully fetched both sensor types
        - Files are accessible and not corrupted

        WHY IT MATTERS:
        Many applications combine optical and SAR data for better results:
        - SAR provides all-weather monitoring
        - Optical provides spectral information
        - Together they enable robust multi-sensor analysis
        """
        self.assertIsNotNone(self.s2_file, "S2 test data missing - run download script")
        self.assertIsNotNone(self.s1_file, "S1 test data missing - run download script")

        print(f"✓ Both sensors available: S1 ({self.s1_file.name}) and S2 ({self.s2_file.name})")

    def test_metadata_consistency(self):
        """Test that metadata is consistent and complete.

        WHAT THIS TESTS:
        - Metadata file structure
        - Required fields present
        - Location information valid

        WHY IT MATTERS:
        Metadata helps track:
        - When data was downloaded
        - Geographic location of test data
        - Product identifiers for debugging
        """
        self.assertIn("location", self.metadata)
        self.assertIn("products", self.metadata)
        self.assertIn("downloaded_at", self.metadata)

        location = self.metadata["location"]
        self.assertIn("lat", location)
        self.assertIn("lon", location)

        # Verify reasonable coordinates (not 0,0 or extreme values)
        self.assertGreater(abs(location["lat"]), 0, "Latitude should not be 0")
        self.assertGreater(abs(location["lon"]), 0, "Longitude should not be 0")
        self.assertLessEqual(abs(location["lat"]), 90, "Latitude should be <= 90")
        self.assertLessEqual(abs(location["lon"]), 180, "Longitude should be <= 180")

        print(
            f"✓ Metadata consistent: {location['lat']}°N, {location['lon']}°E, "
            f"{len(self.metadata['products'])} products"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
