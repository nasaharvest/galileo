"""
Memory-efficient embedding generation for Galileo models.

This module provides a reusable ``make_embeddings`` function that generates
embeddings from satellite imagery without accumulating all batch results in
RAM.  Instead of appending every batch to a Python list and concatenating at
the end (which doubles peak memory), it writes each batch directly into a
**numpy memory-mapped array** (``numpy.memmap``) backed by a temporary file
on disk.

How it works
------------
1. We compute the total number of pixel-batches up front from the spatial
   dimensions of the ``DatasetOutput`` and the requested ``window_size``.
2. We run the **first batch** through the model to discover the embedding
   dimension ``D`` (this avoids hard-coding it).
3. We allocate a ``numpy.memmap`` of shape ``(total_pixels, D)`` in a
   temporary file.  The OS manages paging — only the pages currently being
   written occupy physical RAM.
4. As each subsequent batch completes, its embeddings are written at the
   correct offset in the memmap.  The batch tensor is then freed.
5. After all batches are processed the memmap is reshaped to
   ``(height_batches, width_batches, D)`` and — by default — copied into a
   regular ``numpy.ndarray`` so the caller gets a familiar in-memory object
   and the temp file is cleaned up.  If you pass ``return_memmap=True`` the
   raw memmap is returned instead, keeping memory usage low even after the
   function returns (useful when the result is too large to fit in RAM).

Peak memory
-----------
Approximately ``batch_size × D × 4 bytes`` (one batch of float32 embeddings)
plus whatever the model itself needs.  Compare this with the old approach
which required ``total_pixels × D × 4 bytes`` *twice* (list + concatenation).

Usage example
-------------
.. code-block:: python

    import torch
    from src.inference import make_embeddings
    from src.galileo import Encoder
    from src.data.dataset import Dataset

    # Load your model
    model = Encoder.load_pretrained("path/to/checkpoint")
    model.eval()

    # Load a DatasetOutput (e.g. from a large TIFF)
    ds = Dataset("data/tifs", normalize=True)
    dataset_output = ds.load_tif(0)

    # Generate embeddings — memory-efficient by default
    embeddings = make_embeddings(
        model=model,
        datasetoutput=dataset_output,
        window_size=1,
        patch_size=1,
        batch_size=128,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # embeddings.shape == (H // window_size, W // window_size, D)

    # For truly huge datasets, keep the result on disk:
    embeddings_mmap = make_embeddings(
        model=model,
        datasetoutput=dataset_output,
        window_size=1,
        patch_size=1,
        batch_size=128,
        return_memmap=True,
    )
    # embeddings_mmap is a np.memmap — pages are loaded lazily on access

Parameters
----------
model : torch.nn.Module
    A Galileo ``Encoder`` (or any model whose ``__call__`` returns the same
    tuple structure and exposes ``average_tokens``).
datasetoutput : DatasetOutput
    The satellite image data to embed.  Its spatial dimensions must be
    divisible by ``window_size``.
window_size : int
    Spatial window used by ``DatasetOutput.in_pixel_batches``.
patch_size : int
    Patch size forwarded to the encoder.
batch_size : int, default 128
    Number of pixel windows per forward pass.
device : torch.device | None
    Compute device.  Defaults to CPU.
return_memmap : bool, default False
    If ``True``, return the raw ``numpy.memmap`` instead of copying to a
    regular array.  The caller is then responsible for eventually deleting
    the backing file (available as ``result.filename``).
"""

from __future__ import annotations

import logging
import tempfile
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.masking import MaskedOutput

logger = logging.getLogger(__name__)


def _run_single_batch(
    model: torch.nn.Module,
    batch_datasetoutput: Any,
    device: torch.device,
    patch_size: int,
) -> np.ndarray:
    """Run a single batch through the model and return embeddings as numpy.

    This helper isolates the forward-pass logic so it can be called both for
    the "probe" batch (to discover the embedding dimension) and for every
    subsequent batch.

    Parameters
    ----------
    model : torch.nn.Module
        The Galileo encoder in eval mode.
    batch_datasetoutput : DatasetOutput
        A single batch yielded by ``DatasetOutput.in_pixel_batches``.
    device : torch.device
        The device tensors are moved to before the forward pass.
    patch_size : int
        Patch size forwarded to the encoder.

    Returns
    -------
    np.ndarray
        Embeddings of shape ``(batch_len, D)`` as float32 on CPU.
    """
    masked_output = MaskedOutput.from_datasetoutput(batch_datasetoutput, device=device)
    with torch.no_grad():
        model_output = model(
            masked_output.space_time_x.float(),
            masked_output.space_x.float(),
            masked_output.time_x.float(),
            masked_output.static_x.float(),
            masked_output.space_time_mask,
            masked_output.space_mask,
            torch.ones_like(masked_output.time_mask),
            torch.ones_like(masked_output.static_mask),
            masked_output.months.long(),
            patch_size=patch_size,
        )
        embeddings = model.average_tokens(*model_output[:-1]).cpu().numpy()
    return embeddings


def make_embeddings(
    model: torch.nn.Module,
    datasetoutput: Any,
    window_size: int,
    patch_size: int,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
    return_memmap: bool = False,
) -> np.ndarray:
    """Generate embeddings with constant memory via numpy.memmap.

    This is a drop-in replacement for the old ``make_embeddings`` that
    accumulated batches in a Python list.  See the module docstring for a
    full explanation of the memory-mapping strategy.

    Parameters
    ----------
    model : torch.nn.Module
        A Galileo ``Encoder`` in eval mode.
    datasetoutput : DatasetOutput
        Satellite image data whose spatial dims are divisible by
        ``window_size``.
    window_size : int
        Spatial window for ``in_pixel_batches``.
    patch_size : int
        Patch size forwarded to the encoder.
    batch_size : int, default 128
        Pixels per forward pass.
    device : torch.device | None
        Compute device (defaults to CPU).
    return_memmap : bool, default False
        If True return the raw memmap (disk-backed, lazy-loaded).
        If False (default) copy to a regular ndarray and delete the
        temp file.

    Returns
    -------
    np.ndarray
        Embeddings shaped ``(h_batches, w_batches, embed_dim)``.
        If ``return_memmap=True`` this is actually a ``numpy.memmap``.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()

    # --- Compute total number of pixel-windows ---
    h_b = datasetoutput.space_time_x.shape[0] // window_size
    w_b = datasetoutput.space_time_x.shape[1] // window_size
    total_pixels = h_b * w_b
    logger.info(
        "Embedding grid: %d×%d = %d pixel-windows (window_size=%d)",
        h_b,
        w_b,
        total_pixels,
        window_size,
    )

    # --- Iterator over pixel batches ---
    batch_iter = datasetoutput.in_pixel_batches(batch_size=batch_size, window_size=window_size)

    # --- Probe first batch to discover embedding dimension ---
    first_batch = next(batch_iter)
    first_embeddings = _run_single_batch(model, first_batch, device, patch_size)
    embed_dim = first_embeddings.shape[1]
    logger.info("Embedding dimension: %d (discovered from first batch)", embed_dim)

    # --- Allocate the memory-mapped array ---
    # We use a NamedTemporaryFile so the OS gives us a unique path.
    # delete=False because we need the file to outlive the handle (numpy
    # re-opens it internally).  We clean it up at the end unless the caller
    # asked for the raw memmap.
    tmp = tempfile.NamedTemporaryFile(suffix=".memmap", delete=False)
    tmp_path = tmp.name
    tmp.close()  # close handle; numpy will open it itself

    logger.info("Memmap backing file: %s", tmp_path)

    memmap = np.memmap(
        tmp_path,
        dtype=np.float32,
        mode="w+",
        shape=(total_pixels, embed_dim),
    )

    # --- Write the first batch we already computed ---
    offset = 0
    batch_len = first_embeddings.shape[0]
    memmap[offset : offset + batch_len] = first_embeddings
    offset += batch_len
    batch_count = 1

    # --- Process remaining batches ---
    for batch_datasetoutput in tqdm(batch_iter, desc="Embedding batches"):
        embeddings = _run_single_batch(model, batch_datasetoutput, device, patch_size)
        batch_len = embeddings.shape[0]
        memmap[offset : offset + batch_len] = embeddings
        offset += batch_len
        batch_count += 1

    logger.info("Processed %d batches, wrote %d rows", batch_count, offset)

    # --- Flush to disk and reshape ---
    memmap.flush()

    # Reshape to (h_b, w_b, D) — this is a view, no copy
    reshaped = memmap.reshape((h_b, w_b, embed_dim))

    if return_memmap:
        # Caller takes ownership of the temp file
        logger.info("Returning memmap (caller owns %s)", tmp_path)
        return reshaped

    # Default: copy into a regular ndarray and clean up the temp file
    result = np.array(reshaped)
    del memmap
    del reshaped
    import os

    try:
        os.unlink(tmp_path)
        logger.debug("Deleted temp memmap file %s", tmp_path)
    except OSError:
        logger.warning("Could not delete temp file %s", tmp_path)

    return result
