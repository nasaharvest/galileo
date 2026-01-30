"""Basic tests for Copernicus S2 client functionality.

These tests verify core functionality without making real API calls:
- OAuth token generation (mocked)
- Input validation (bbox, dates)
- Cache key generation
- S2 search query construction
"""

import unittest
from unittest.mock import Mock, patch

from src.data.copernicus.client import CopernicusClient
from src.data.copernicus.utils import (
    bbox_to_wkt,
    build_cache_key,
    sanitize_filename,
    validate_bbox,
    validate_date,
    validate_date_range,
)


class TestCopernicusUtils(unittest.TestCase):
    """Test utility functions."""

    def test_validate_bbox_valid(self):
        """Test that valid bboxes pass validation."""
        valid_bboxes = [
            [0, 0, 1, 1],
            [-180, -90, 180, 90],
            [25.6796, -27.6721, 25.6897, -27.663],
        ]
        for bbox in valid_bboxes:
            validate_bbox(bbox)  # Should not raise

    def test_validate_bbox_invalid_format(self):
        """Test that invalid bbox formats raise ValueError."""
        invalid_bboxes = [
            [0, 0, 1],  # Too few elements
            [0, 0, 1, 1, 1],  # Too many elements
            "not a list",  # Wrong type
        ]
        for bbox in invalid_bboxes:
            with self.assertRaises(ValueError):
                validate_bbox(bbox)

    def test_validate_bbox_invalid_coordinates(self):
        """Test that invalid coordinates raise ValueError."""
        invalid_bboxes = [
            [-181, 0, 0, 1],  # Longitude out of range
            [0, -91, 1, 0],  # Latitude out of range
            [1, 0, 0, 1],  # min_lon >= max_lon
            [0, 1, 1, 0],  # min_lat >= max_lat
        ]
        for bbox in invalid_bboxes:
            with self.assertRaises(ValueError):
                validate_bbox(bbox)

    def test_validate_date_valid(self):
        """Test that valid dates are parsed correctly."""
        valid_dates = ["2022-01-01", "2023-12-31", "2024-02-29"]  # Leap year
        for date_str in valid_dates:
            dt = validate_date(date_str)
            self.assertIsNotNone(dt)

    def test_validate_date_invalid(self):
        """Test that invalid dates raise ValueError."""
        invalid_dates = [
            "2022-13-01",  # Invalid month
            "2022-02-30",  # Invalid day
            "2022/01/01",  # Wrong format
            "not a date",  # Completely wrong
        ]
        for date_str in invalid_dates:
            with self.assertRaises(ValueError):
                validate_date(date_str)

    def test_validate_date_range_valid(self):
        """Test that valid date ranges pass validation."""
        start_dt, end_dt = validate_date_range("2022-01-01", "2022-12-31")
        self.assertIsNotNone(start_dt)
        self.assertIsNotNone(end_dt)
        self.assertLess(start_dt, end_dt)

    def test_validate_date_range_invalid(self):
        """Test that invalid date ranges raise ValueError."""
        with self.assertRaises(ValueError):
            validate_date_range("2022-12-31", "2022-01-01")  # start >= end

    def test_bbox_to_wkt(self):
        """Test WKT polygon generation from bbox."""
        bbox = [0, 0, 1, 1]
        wkt = bbox_to_wkt(bbox)
        self.assertIn("POLYGON", wkt)
        self.assertIn("0 0", wkt)
        self.assertIn("1 1", wkt)

    def test_build_cache_key_deterministic(self):
        """Test that cache keys are deterministic."""
        params = {"bbox": [0, 0, 1, 1], "start_date": "2022-01-01", "end_date": "2022-12-31"}
        key1 = build_cache_key("s2", **params)
        key2 = build_cache_key("s2", **params)
        self.assertEqual(key1, key2)

    def test_build_cache_key_different_params(self):
        """Test that different parameters produce different cache keys."""
        key1 = build_cache_key("s2", bbox=[0, 0, 1, 1])
        key2 = build_cache_key("s2", bbox=[0, 0, 2, 2])
        self.assertNotEqual(key1, key2)

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        unsafe = "S2A_MSIL1C_20220101T123456:789_N0400.SAFE"
        safe = sanitize_filename(unsafe)
        self.assertNotIn(":", safe)
        self.assertIn("_", safe)


class TestCopernicusClient(unittest.TestCase):
    """Test CopernicusClient initialization and authentication."""

    @patch.dict(
        "os.environ",
        {"COPERNICUS_CLIENT_ID": "test_id", "COPERNICUS_CLIENT_SECRET": "test_secret"},
    )
    def test_client_init_from_env(self):
        """Test client initialization from environment variables."""
        client = CopernicusClient(load_dotenv_file=False)
        self.assertEqual(client.client_id, "test_id")
        self.assertEqual(client.client_secret, "test_secret")

    def test_client_init_from_params(self):
        """Test client initialization from parameters."""
        client = CopernicusClient(
            load_dotenv_file=False,
            client_id="param_id",
            client_secret="param_secret",
        )
        self.assertEqual(client.client_id, "param_id")
        self.assertEqual(client.client_secret, "param_secret")

    @patch.dict("os.environ", {}, clear=True)
    def test_client_init_missing_credentials(self):
        """Test that missing credentials raise ValueError."""
        with self.assertRaises(ValueError):
            CopernicusClient(load_dotenv_file=False)

    @patch.dict(
        "os.environ",
        {"COPERNICUS_CLIENT_ID": "test_id", "COPERNICUS_CLIENT_SECRET": "test_secret"},
    )
    @patch("src.data.copernicus.client.requests.Session")
    def test_get_access_token(self, mock_session_class):
        """Test OAuth token generation."""
        # Mock the session and response
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "test_token",
            "expires_in": 3600,
        }
        mock_session.post.return_value = mock_response

        client = CopernicusClient(load_dotenv_file=False)
        token = client._get_access_token()

        self.assertEqual(token, "test_token")
        self.assertIsNotNone(client._access_token)
        self.assertGreater(client._token_expires_at, 0)

    @patch.dict(
        "os.environ",
        {"COPERNICUS_CLIENT_ID": "test_id", "COPERNICUS_CLIENT_SECRET": "test_secret"},
    )
    def test_fetch_s2_validation(self):
        """Test that fetch_s2 validates input parameters."""
        client = CopernicusClient(load_dotenv_file=False)

        # Invalid bbox
        with self.assertRaises(ValueError):
            client.fetch_s2(
                bbox=[0, 0, 1],  # Too few elements
                start_date="2022-01-01",
                end_date="2022-12-31",
            )

        # Invalid resolution
        with self.assertRaises(ValueError):
            client.fetch_s2(
                bbox=[0, 0, 1, 1],
                start_date="2022-01-01",
                end_date="2022-12-31",
                resolution=15,  # Invalid resolution
            )

        # Invalid cloud cover
        with self.assertRaises(ValueError):
            client.fetch_s2(
                bbox=[0, 0, 1, 1],
                start_date="2022-01-01",
                end_date="2022-12-31",
                max_cloud_cover=150,  # Out of range
            )

        # Invalid date range
        with self.assertRaises(ValueError):
            client.fetch_s2(
                bbox=[0, 0, 1, 1],
                start_date="2022-12-31",
                end_date="2022-01-01",  # End before start
            )


class TestS2SearchQuery(unittest.TestCase):
    """Test S2 search query construction."""

    @patch.dict(
        "os.environ",
        {"COPERNICUS_CLIENT_ID": "test_id", "COPERNICUS_CLIENT_SECRET": "test_secret"},
    )
    @patch("src.data.copernicus.client.requests.Session")
    @patch("src.data.copernicus.s2._search_s2_products")
    def test_search_query_construction(self, mock_search, mock_session_class):
        """Test that S2 search queries are constructed correctly."""
        # Mock the session
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock search to return empty list
        mock_search.return_value = []

        client = CopernicusClient(load_dotenv_file=False)

        # Call fetch_s2 with specific parameters
        client.fetch_s2(
            bbox=[25.6796, -27.6721, 25.6897, -27.663],
            start_date="2022-01-01",
            end_date="2022-01-31",
            resolution=10,
            max_cloud_cover=20.0,
            product_type="S2MSI2A",
            download_data=False,
            interactive=False,
        )

        # Verify search was called with correct parameters
        mock_search.assert_called_once()
        call_args = mock_search.call_args

        # call_args[0] contains positional args, call_args[1] contains kwargs
        # The function is called with positional args
        self.assertEqual(call_args[0][1], [25.6796, -27.6721, 25.6897, -27.663])  # bbox
        self.assertEqual(call_args[0][2], "2022-01-01")  # start_date
        self.assertEqual(call_args[0][3], "2022-01-31")  # end_date
        self.assertEqual(call_args[0][4], 20.0)  # max_cloud_cover
        self.assertEqual(call_args[0][5], "S2MSI2A")  # product_type


class TestCaching(unittest.TestCase):
    """Test caching behavior."""

    @patch.dict(
        "os.environ",
        {"COPERNICUS_CLIENT_ID": "test_id", "COPERNICUS_CLIENT_SECRET": "test_secret"},
    )
    def test_cache_key_includes_all_params(self):
        """Test that cache keys include all relevant parameters."""

        # Create two cache keys with different parameters
        key1 = build_cache_key(
            "s2",
            bbox=[0, 0, 1, 1],
            start_date="2022-01-01",
            end_date="2022-12-31",
            resolution=10,
            max_cloud_cover=20.0,
            product_type="S2MSI1C",
            download_data=True,
        )

        key2 = build_cache_key(
            "s2",
            bbox=[0, 0, 1, 1],
            start_date="2022-01-01",
            end_date="2022-12-31",
            resolution=20,  # Different resolution
            max_cloud_cover=20.0,
            product_type="S2MSI1C",
            download_data=True,
        )

        # Keys should be different
        self.assertNotEqual(key1, key2)


if __name__ == "__main__":
    unittest.main()
