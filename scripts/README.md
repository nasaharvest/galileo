# Test Data Scripts

## Overview

This directory contains scripts for managing test data for the Copernicus client integration tests.

## Download Test Data

The `download_test_data.py` script downloads real Sentinel-1 and Sentinel-2 products to use as test fixtures.

### Prerequisites

1. Copernicus Data Space credentials (free registration at https://dataspace.copernicus.eu/)
2. Credentials in `.env` file:
   ```bash
   COPERNICUS_CLIENT_ID=your_client_id
   COPERNICUS_CLIENT_SECRET=your_client_secret
   ```
3. ~3-4 GB free disk space

### Usage

```bash
# Download test data (run once)
uv run python scripts/download_test_data.py
```

This will:
- Download 2 Sentinel-1 products (~1.2-1.7 GB each)
- Download 1 Sentinel-2 product (~500-800 MB)
- Save products to `data/cache/copernicus/s1/` and `data/cache/copernicus/s2/`
- Create metadata file `data/test_fixtures/test_data_metadata.json`

### Test Location

The script downloads data for an agricultural area in the Netherlands:
- Location: 52.0°N, 5.5°E
- S2 Area: ~5km × 5km
- S1 Area: ~100km × 100km (larger due to S1 swath width)
- Time period: January-July 2024

This location was chosen because:
- Agricultural area (good for testing NDVI, crop monitoring)
- Reliable S1/S2 coverage
- Good data availability
- Flat terrain (easier to process)

### Running Tests

Once test data is downloaded, run the integration tests:

```bash
# Run all integration tests
uv run python -m pytest tests/test_copernicus_integration.py -v

# Run specific test
uv run python -m pytest tests/test_copernicus_integration.py::TestCopernicusIntegration::test_s2_rgb_extraction -v
```

### Troubleshooting

**No products found:**
- Try different dates (the script uses January-July 2024)
- Check cloud cover threshold for S2
- Verify location has satellite coverage

**Download fails:**
- Check credentials in `.env` (use CLIENT_ID and CLIENT_SECRET, not USERNAME/PASSWORD)
- Verify internet connection
- Check Copernicus Data Space status
- Large files may take 1-2 minutes each

**Tests skip:**
- Run `download_test_data.py` first
- Check that files exist in `data/cache/copernicus/s1/` and `data/cache/copernicus/s2/`
- Verify `test_data_metadata.json` exists

### Updating Test Data

To download fresh test data:

```bash
# Remove old cache
rm -rf data/cache/copernicus/

# Download new data
uv run python scripts/download_test_data.py
```

### Data Management

Test fixtures are gitignored (via `data/*` in `.gitignore`). Each developer downloads their own test data locally.

**Disk usage:**
- S1 products: ~1.2-1.7 GB each (2 products = ~3 GB)
- S2 product: ~500-800 MB
- Total: ~3.5-4.5 GB

To clean up:
```bash
rm -rf data/cache/copernicus/
rm -rf data/test_fixtures/
```
