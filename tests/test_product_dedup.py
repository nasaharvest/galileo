"""Tests for product-level deduplication in Copernicus downloads.

Verifies that:
- find_product_on_disk correctly detects already-downloaded products by UUID
- process_products skips downloads when a product is already on disk
- Different queries returning the same product ID share the download
- Edge cases: empty files, missing dirs, no product ID, corrupted zips
"""

import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List
from unittest.mock import MagicMock

from src.data.copernicus.common import find_product_on_disk, process_products

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRODUCT_UUID_A = "a8dd0899-7a3b-4e4b-9b3a-5e7f1234abcd"
PRODUCT_UUID_B = "b7cc1234-5d6e-4f7a-8b9c-0d1e2f3a4b5c"
PRODUCT_UUID_C = "c6bb5678-9a0b-1c2d-3e4f-567890abcdef"


def _make_product(product_id: str, name: str, size: int = 1000) -> Dict[str, Any]:
    """Create a fake Copernicus product dict matching the API shape."""
    return {
        "Id": product_id,
        "Name": name,
        "ContentLength": size,
        "ContentDate": {"Start": "2022-01-01T00:00:00.000Z"},
    }


def _create_fake_file(directory: Path, filename: str, size: int = 100) -> Path:
    """Create a fake file with some content.

    For .zip files, creates a valid zip archive so it passes integrity checks.
    For other files, writes raw bytes.
    """
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if filename.endswith(".zip"):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("placeholder.txt", "x" * size)
    else:
        path.write_bytes(b"x" * size)
    return path


def _make_mock_client(cache_dir: Path) -> MagicMock:
    """Create a mock CopernicusClient with the given cache_dir."""
    client = MagicMock()
    client.cache_dir = cache_dir
    return client


# ---------------------------------------------------------------------------
# find_product_on_disk
# ---------------------------------------------------------------------------


class TestFindProductOnDisk(unittest.TestCase):
    """Tests for the find_product_on_disk function."""

    def test_finds_existing_zip(self):
        """Product zip with embedded UUID is found."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            _create_fake_file(
                cache / "s2",
                f"{PRODUCT_UUID_A}__S2A_MSIL1C_20220101_R10m.zip",
            )
            result = find_product_on_disk(cache, "s2", PRODUCT_UUID_A)
            self.assertIsNotNone(result)
            self.assertIn(PRODUCT_UUID_A, result.name)

    def test_finds_existing_metadata_json(self):
        """Metadata json with embedded UUID is found."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            _create_fake_file(
                cache / "s1",
                f"{PRODUCT_UUID_B}__S1A_IW_GRDH_metadata.json",
            )
            result = find_product_on_disk(cache, "s1", PRODUCT_UUID_B)
            self.assertIsNotNone(result)

    def test_returns_none_when_not_present(self):
        """Returns None when no file matches the product ID."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "s2").mkdir(parents=True)
            # Create a file with a DIFFERENT product ID
            _create_fake_file(
                cache / "s2",
                f"{PRODUCT_UUID_A}__some_product.zip",
            )
            result = find_product_on_disk(cache, "s2", PRODUCT_UUID_B)
            self.assertIsNone(result)

    def test_returns_none_when_subdir_missing(self):
        """Returns None when the satellite subdirectory doesn't exist."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            result = find_product_on_disk(cache, "s2", PRODUCT_UUID_A)
            self.assertIsNone(result)

    def test_ignores_empty_files(self):
        """Empty files (0 bytes) are not considered valid downloads."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            path = cache / "s2" / f"{PRODUCT_UUID_A}__S2A_product.zip"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"")  # 0 bytes
            result = find_product_on_disk(cache, "s2", PRODUCT_UUID_A)
            self.assertIsNone(result)

    def test_detects_corrupted_zip(self):
        """A truncated/corrupted zip is rejected and deleted."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            path = cache / "s2" / f"{PRODUCT_UUID_A}__S2A_product.zip"
            path.parent.mkdir(parents=True)
            # Write garbage that isn't a valid zip
            path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)  # zip magic + junk
            result = find_product_on_disk(cache, "s2", PRODUCT_UUID_A)
            self.assertIsNone(result)
            # The corrupted file should have been cleaned up
            self.assertFalse(path.exists())

    def test_accepts_valid_zip(self):
        """A valid zip file is accepted."""
        import zipfile as zf

        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            path = cache / "s2" / f"{PRODUCT_UUID_A}__S2A_product.zip"
            path.parent.mkdir(parents=True)
            # Create a real valid zip
            with zf.ZipFile(path, "w") as z:
                z.writestr("test.txt", "hello")
            result = find_product_on_disk(cache, "s2", PRODUCT_UUID_A)
            self.assertIsNotNone(result)

    def test_non_zip_files_skip_zip_check(self):
        """Non-zip files (like metadata json) only need size > 0 check."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            path = cache / "s1" / f"{PRODUCT_UUID_A}__S1A_metadata.json"
            path.parent.mkdir(parents=True)
            path.write_bytes(b'{"product_id": "test"}')
            result = find_product_on_disk(cache, "s1", PRODUCT_UUID_A)
            self.assertIsNotNone(result)

    def test_does_not_match_partial_uuid(self):
        """A file whose name starts with a prefix of the UUID should not match."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            # UUID_A starts with "a8dd0899-..."
            # Create a file with a truncated UUID — glob should NOT match
            _create_fake_file(cache / "s2", "a8dd0899__truncated.zip")
            result = find_product_on_disk(cache, "s2", PRODUCT_UUID_A)
            self.assertIsNone(result)

    def test_correct_subdir_isolation(self):
        """A product in s1/ is not found when searching s2/."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            _create_fake_file(
                cache / "s1",
                f"{PRODUCT_UUID_A}__S1A_product.zip",
            )
            result_s2 = find_product_on_disk(cache, "s2", PRODUCT_UUID_A)
            result_s1 = find_product_on_disk(cache, "s1", PRODUCT_UUID_A)
            self.assertIsNone(result_s2)
            self.assertIsNotNone(result_s1)


# ---------------------------------------------------------------------------
# process_products — dedup integration
# ---------------------------------------------------------------------------


class TestProcessProductsDedup(unittest.TestCase):
    """Tests that process_products skips already-downloaded products."""

    def test_skips_download_when_product_on_disk(self):
        """download_func should NOT be called for a product already on disk."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            client = _make_mock_client(cache)

            # Pre-create a file on disk for product A
            existing_file = _create_fake_file(
                cache / "s2",
                f"{PRODUCT_UUID_A}__S2A_MSIL1C_R10m.zip",
            )

            products = [_make_product(PRODUCT_UUID_A, "S2A_MSIL1C_20220101")]
            download_func = MagicMock(return_value=None)
            metadata_func = MagicMock(return_value=None)

            paths = process_products(
                client=client,
                products=products,
                download_data=True,
                satellite="SENTINEL-2",
                download_func=download_func,
                metadata_func=metadata_func,
            )

            # download_func should never have been called
            download_func.assert_not_called()
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0], existing_file)

    def test_downloads_when_product_not_on_disk(self):
        """download_func IS called when the product is not on disk."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "s2").mkdir(parents=True)
            client = _make_mock_client(cache)

            new_file = cache / "s2" / f"{PRODUCT_UUID_A}__new.zip"
            new_file.write_bytes(b"data")

            products = [_make_product(PRODUCT_UUID_A, "S2A_MSIL1C_20220101")]
            download_func = MagicMock(return_value=new_file)
            metadata_func = MagicMock(return_value=None)

            # No pre-existing file — download_func should be called
            # But first remove the file we just created so dedup doesn't find it
            new_file.unlink()

            paths = process_products(
                client=client,
                products=products,
                download_data=True,
                satellite="SENTINEL-2",
                download_func=download_func,
                metadata_func=metadata_func,
            )

            download_func.assert_called_once()
            self.assertEqual(len(paths), 1)

    def test_mixed_skip_and_download(self):
        """With 3 products, 1 already on disk and 2 new, only 2 downloads happen."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            client = _make_mock_client(cache)

            # Product A is already on disk
            existing = _create_fake_file(
                cache / "s2",
                f"{PRODUCT_UUID_A}__S2A_existing.zip",
            )

            products = [
                _make_product(PRODUCT_UUID_A, "S2A_existing"),
                _make_product(PRODUCT_UUID_B, "S2A_new_b"),
                _make_product(PRODUCT_UUID_C, "S2A_new_c"),
            ]

            call_count = 0

            def fake_download(client, product, index, **kwargs):
                nonlocal call_count
                call_count += 1
                pid = product["Id"]
                path = cache / "s2" / f"{pid}__downloaded.zip"
                with zipfile.ZipFile(path, "w") as zf:
                    zf.writestr("data.txt", "satellite-data")
                return path

            paths = process_products(
                client=client,
                products=products,
                download_data=True,
                satellite="SENTINEL-2",
                download_func=fake_download,
                metadata_func=MagicMock(),
            )

            self.assertEqual(call_count, 2)  # Only B and C downloaded
            self.assertEqual(len(paths), 3)  # All 3 returned
            self.assertEqual(paths[0], existing)  # A was the cached one

    def test_skips_metadata_when_product_on_disk(self):
        """metadata_func should NOT be called for a product already on disk."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            client = _make_mock_client(cache)

            existing = _create_fake_file(
                cache / "s1",
                f"{PRODUCT_UUID_A}__S1A_metadata.json",
            )

            products = [_make_product(PRODUCT_UUID_A, "S1A_IW_GRDH")]
            download_func = MagicMock()
            metadata_func = MagicMock(return_value=None)

            paths = process_products(
                client=client,
                products=products,
                download_data=False,
                satellite="SENTINEL-1",
                download_func=download_func,
                metadata_func=metadata_func,
            )

            metadata_func.assert_not_called()
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0], existing)

    def test_no_dedup_when_product_has_no_id(self):
        """Products without an Id field should always go through download."""
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "s2").mkdir(parents=True)
            client = _make_mock_client(cache)

            product_no_id = {"Name": "S2A_unknown", "ContentLength": 100}
            dummy_path = cache / "s2" / "downloaded.zip"
            dummy_path.write_bytes(b"data")

            download_func = MagicMock(return_value=dummy_path)

            paths = process_products(
                client=client,
                products=[product_no_id],
                download_data=True,
                satellite="SENTINEL-2",
                download_func=download_func,
                metadata_func=MagicMock(),
            )

            download_func.assert_called_once()
            self.assertEqual(len(paths), 1)


# ---------------------------------------------------------------------------
# Scenario: two different bboxes, same tile
# ---------------------------------------------------------------------------


class TestCrossBboxDedup(unittest.TestCase):
    """Simulate the real scenario: two queries with different bboxes return the same product."""

    def test_second_query_skips_download(self):
        """
        Query 1 (bbox_a) downloads product X.
        Query 2 (bbox_b, slightly shifted) returns the same product X.
        Product X should NOT be downloaded again.
        """
        with TemporaryDirectory() as tmp:
            cache = Path(tmp)
            client = _make_mock_client(cache)

            product_x = _make_product(PRODUCT_UUID_A, "S2A_MSIL1C_20220101_T31UGQ")
            download_calls: List[str] = []

            def fake_download(client, product, index, **kwargs):
                pid = product["Id"]
                download_calls.append(pid)
                path = cache / "s2" / f"{pid}__S2A_MSIL1C_20220101_T31UGQ_R10m.zip"
                path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(path, "w") as zf:
                    zf.writestr("data.txt", "satellite-data")
                return path

            # --- Query 1: first time, product gets downloaded ---
            paths_1 = process_products(
                client=client,
                products=[product_x],
                download_data=True,
                satellite="SENTINEL-2",
                download_func=fake_download,
                metadata_func=MagicMock(),
            )
            self.assertEqual(len(download_calls), 1)
            self.assertEqual(len(paths_1), 1)

            # --- Query 2: same product returned, should be skipped ---
            paths_2 = process_products(
                client=client,
                products=[product_x],
                download_data=True,
                satellite="SENTINEL-2",
                download_func=fake_download,
                metadata_func=MagicMock(),
            )
            # download_func should NOT have been called again
            self.assertEqual(len(download_calls), 1)
            self.assertEqual(len(paths_2), 1)
            # Both queries return the same file
            self.assertEqual(paths_1[0], paths_2[0])


if __name__ == "__main__":
    unittest.main()
