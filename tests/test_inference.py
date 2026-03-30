"""
Tests for src.inference — memory-efficient embedding generation.

These tests verify that ``make_embeddings`` produces correct results while
using numpy.memmap under the hood.  We mock the Galileo encoder so the tests
run fast and without a GPU or real model weights.

Test strategy
-------------
* **Fake model**: A lightweight ``FakeEncoder`` that returns deterministic
  embeddings (the mean of each pixel-window's space_time_x values, broadcast
  to a fixed embedding dimension).  This lets us assert exact numerical
  results without loading real weights.
* **Synthetic DatasetOutput**: Small arrays with known values so we can
  predict the expected output.
* **Memmap path**: We verify that the ``return_memmap=True`` code path
  returns a ``numpy.memmap`` backed by a real file, and that the default
  path returns a plain ``numpy.ndarray`` with no leftover temp file.
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from src.data.dataset import DatasetOutput

# ---------------------------------------------------------------------------
# Fake encoder that mimics the Galileo Encoder interface
# ---------------------------------------------------------------------------

FAKE_EMBED_DIM = 8  # small for fast tests


class FakeEncoder(torch.nn.Module):
    """A minimal stand-in for ``src.galileo.Encoder``.

    The forward pass ignores the actual input content and returns a tuple of
    tensors whose structure matches what ``Encoder.__call__`` returns.  The
    ``average_tokens`` class method is also faked to return a deterministic
    embedding: for each sample in the batch it returns a vector of shape
    ``(FAKE_EMBED_DIM,)`` filled with the sample's index in the batch.
    This makes assertions trivial.
    """

    def __init__(self):
        super().__init__()
        # Need at least one parameter so model.eval() works
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months, patch_size):
        batch_size = s_t_x.shape[0]
        device = s_t_x.device
        # Return a tuple of 5 elements (4 used by average_tokens + 1 extra)
        # Each is (batch, D) — the exact shapes don't matter because
        # average_tokens is also faked.
        dummy = torch.zeros(batch_size, FAKE_EMBED_DIM, device=device)
        # We stash the batch size so average_tokens can use it
        self._last_batch_size = batch_size
        self._last_device = device
        return (dummy, dummy, dummy, dummy, dummy)

    @classmethod
    def average_tokens(cls, s_t_x, sp_x, t_x, st_x):
        """Return deterministic embeddings: sample i → vector filled with i."""
        batch_size = s_t_x.shape[0]
        device = s_t_x.device
        # Each sample gets a unique but predictable embedding
        indices = torch.arange(batch_size, dtype=torch.float32, device=device)
        return indices.unsqueeze(1).expand(batch_size, FAKE_EMBED_DIM)


# ---------------------------------------------------------------------------
# Helper to build a synthetic DatasetOutput
# ---------------------------------------------------------------------------


def _make_dataset_output(height: int, width: int, timesteps: int = 3) -> DatasetOutput:
    """Create a small DatasetOutput with known shapes.

    Parameters
    ----------
    height, width : int
        Spatial dimensions (must be compatible with the window_size used in
        the test).
    timesteps : int
        Number of time steps.

    Returns
    -------
    DatasetOutput
        With random but reproducible data.
    """
    rng = np.random.RandomState(42)

    # Number of bands — taken from the real Galileo config
    n_space_time_bands = 18  # len(SPACE_TIME_BANDS)
    n_space_bands = 2  # len(SPACE_BANDS)
    n_time_bands = 1  # len(TIME_BANDS)
    n_static_bands = 1  # len(STATIC_BANDS)

    return DatasetOutput(
        space_time_x=rng.randn(height, width, timesteps, n_space_time_bands).astype(np.float32),
        space_x=rng.randn(height, width, n_space_bands).astype(np.float32),
        time_x=rng.randn(timesteps, n_time_bands).astype(np.float32),
        static_x=rng.randn(n_static_bands).astype(np.float32),
        months=np.array([1, 2, 3][:timesteps], dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestMakeEmbeddings(unittest.TestCase):
    """Tests for ``src.inference.make_embeddings``."""

    def setUp(self):
        self.model = FakeEncoder()
        self.device = torch.device("cpu")

    # -- Shape tests --------------------------------------------------------

    def test_output_shape_basic(self):
        """Output shape should be (H/window, W/window, embed_dim)."""
        from src.inference import make_embeddings

        height, width = 4, 6
        window_size = 1
        ds = _make_dataset_output(height, width)

        result = make_embeddings(
            self.model,
            ds,
            window_size=window_size,
            patch_size=1,
            batch_size=4,
            device=self.device,
        )

        self.assertEqual(result.shape, (height, width, FAKE_EMBED_DIM))

    def test_output_shape_with_window(self):
        """With window_size=2 the spatial dims should halve."""
        from src.inference import make_embeddings

        height, width = 4, 6
        window_size = 2
        ds = _make_dataset_output(height, width)

        result = make_embeddings(
            self.model,
            ds,
            window_size=window_size,
            patch_size=1,
            batch_size=8,
            device=self.device,
        )

        self.assertEqual(result.shape, (height // 2, width // 2, FAKE_EMBED_DIM))

    # -- Return type tests --------------------------------------------------

    def test_default_returns_ndarray(self):
        """By default the result should be a plain numpy.ndarray, not a memmap."""
        from src.inference import make_embeddings

        ds = _make_dataset_output(4, 4)
        result = make_embeddings(
            self.model,
            ds,
            window_size=1,
            patch_size=1,
            batch_size=8,
            device=self.device,
        )

        self.assertIsInstance(result, np.ndarray)
        # Specifically NOT a memmap
        self.assertNotIsInstance(result, np.memmap)

    def test_return_memmap_flag(self):
        """With return_memmap=True the result should be a numpy.memmap."""
        from src.inference import make_embeddings

        ds = _make_dataset_output(4, 4)
        result = make_embeddings(
            self.model,
            ds,
            window_size=1,
            patch_size=1,
            batch_size=8,
            device=self.device,
            return_memmap=True,
        )

        self.assertIsInstance(result, np.memmap)
        # The backing file should exist
        self.assertTrue(os.path.exists(result.filename))

        # Clean up
        filename = result.filename
        del result
        os.unlink(filename)

    # -- Numerical correctness ----------------------------------------------

    def test_values_match_sequential_processing(self):
        """Memmap result should be identical to processing batches in a list.

        We run the same data through both the old list-based approach and the
        new memmap approach and compare element-wise.
        """
        from src.inference import _run_single_batch, make_embeddings

        height, width = 6, 4
        ds = _make_dataset_output(height, width)
        window_size = 1
        batch_size = 5  # intentionally not a divisor of 24

        # --- Old-style list accumulation ---
        self.model.eval()
        old_list = []
        for batch in ds.in_pixel_batches(batch_size=batch_size, window_size=window_size):
            emb = _run_single_batch(self.model, batch, self.device, patch_size=1)
            old_list.append(emb)
        old_result = np.concatenate(old_list, axis=0).reshape(height, width, FAKE_EMBED_DIM)

        # --- New memmap approach ---
        new_result = make_embeddings(
            self.model,
            ds,
            window_size=window_size,
            patch_size=1,
            batch_size=batch_size,
            device=self.device,
        )

        np.testing.assert_array_equal(old_result, new_result)

    # -- Batch size edge cases ----------------------------------------------

    def test_batch_size_one(self):
        """Should work even with batch_size=1 (many batches)."""
        from src.inference import make_embeddings

        ds = _make_dataset_output(2, 3)
        result = make_embeddings(
            self.model,
            ds,
            window_size=1,
            patch_size=1,
            batch_size=1,
            device=self.device,
        )
        self.assertEqual(result.shape, (2, 3, FAKE_EMBED_DIM))

    def test_batch_size_larger_than_total(self):
        """Should work when batch_size >= total pixels (single batch)."""
        from src.inference import make_embeddings

        ds = _make_dataset_output(2, 3)
        result = make_embeddings(
            self.model,
            ds,
            window_size=1,
            patch_size=1,
            batch_size=1000,
            device=self.device,
        )
        self.assertEqual(result.shape, (2, 3, FAKE_EMBED_DIM))

    # -- Temp file cleanup --------------------------------------------------

    def test_temp_file_cleaned_up(self):
        """When return_memmap=False the temp file should be deleted."""
        from src.inference import make_embeddings

        # We'll spy on tempfile to capture the path
        ds = _make_dataset_output(2, 2)

        # Count .memmap files in temp dir before and after
        tmp_dir = tempfile.gettempdir()
        before = {f for f in os.listdir(tmp_dir) if f.endswith(".memmap")}

        _ = make_embeddings(
            self.model,
            ds,
            window_size=1,
            patch_size=1,
            batch_size=4,
            device=self.device,
        )

        after = {f for f in os.listdir(tmp_dir) if f.endswith(".memmap")}
        new_files = after - before
        self.assertEqual(len(new_files), 0, f"Leftover memmap files: {new_files}")

    # -- Device default -----------------------------------------------------

    def test_default_device_is_cpu(self):
        """When device=None it should default to CPU without error."""
        from src.inference import make_embeddings

        ds = _make_dataset_output(2, 2)
        result = make_embeddings(
            self.model,
            ds,
            window_size=1,
            patch_size=1,
            batch_size=4,
            device=None,
        )
        self.assertEqual(result.shape, (2, 2, FAKE_EMBED_DIM))


if __name__ == "__main__":
    unittest.main()
