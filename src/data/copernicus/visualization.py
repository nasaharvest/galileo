"""Visualization utilities for Copernicus Sentinel-2 satellite data.

This module provides high-level plotting functions for displaying
Sentinel-2 optical imagery, coverage maps, and analysis results.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .image_processing import extract_rgb_composite, get_image_statistics


def create_coverage_map(
    target_bbox: List[float],
    center_lat: float,
    center_lon: float,
    s2_files: List[Path],
    ax: Optional[plt.Axes] = None,
    title: str = "Satellite Coverage Map",
) -> plt.Axes:
    """Create a coverage map showing target area and available products.

    Args:
        target_bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84
        center_lat: Center latitude
        center_lon: Center longitude
        s2_files: List of available Sentinel-2 files
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Map extent with padding
    padding = 0.01
    map_extent = [
        target_bbox[0] - padding,
        target_bbox[2] + padding,
        target_bbox[1] - padding,
        target_bbox[3] + padding,
    ]

    # Create coordinate grid for background
    lons = np.linspace(map_extent[0], map_extent[1], 100)
    lats = np.linspace(map_extent[2], map_extent[3], 100)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Background terrain pattern
    elevation = np.sin(lon_grid * 100) * np.cos(lat_grid * 100) * 0.1
    ax.contourf(lon_grid, lat_grid, elevation, levels=20, cmap="terrain", alpha=0.3)

    # Plot target bounding box
    bbox_lons = [target_bbox[0], target_bbox[2], target_bbox[2], target_bbox[0], target_bbox[0]]
    bbox_lats = [target_bbox[1], target_bbox[1], target_bbox[3], target_bbox[3], target_bbox[1]]
    ax.plot(bbox_lons, bbox_lats, "r-", linewidth=3, label="Target Area")
    ax.fill(bbox_lons, bbox_lats, "red", alpha=0.2)

    # Center point
    ax.plot(center_lon, center_lat, "ro", markersize=10, label="Center Point")

    # Satellite coverage
    if s2_files:
        coverage_lons = [
            target_bbox[0] - 0.005,
            target_bbox[2] + 0.005,
            target_bbox[2] + 0.005,
            target_bbox[0] - 0.005,
            target_bbox[0] - 0.005,
        ]
        coverage_lats = [
            target_bbox[1] - 0.005,
            target_bbox[1] - 0.005,
            target_bbox[3] + 0.005,
            target_bbox[3] + 0.005,
            target_bbox[1] - 0.005,
        ]
        ax.plot(
            coverage_lons,
            coverage_lats,
            "b--",
            linewidth=2,
            alpha=0.7,
            label=f"Copernicus Products ({len(s2_files)} found)",
        )

    # Customize plot
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Add info box
    area_km = ((target_bbox[2] - target_bbox[0]) * 111.32 * np.cos(np.radians(center_lat))) * (
        (target_bbox[3] - target_bbox[1]) * 110.54
    )

    info_text = (
        f"Target Area:\n"
        f"• Lat: {center_lat:.4f}°N\n"
        f"• Lon: {center_lon:.4f}°E\n"
        f"• Area: ~{area_km:.1f} km²\n"
        f"• Products: {len(s2_files)}"
    )

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    return ax


def display_satellite_image(
    zip_file_path: Path,
    target_bbox: List[float],
    ax: Optional[plt.Axes] = None,
    bands: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> Optional[plt.Axes]:
    """Display satellite image from ZIP file with target area overlay.

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        target_bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84
        ax: Matplotlib axes (creates new if None)
        bands: Band names for composite (default: RGB)
        title: Plot title (auto-generated if None)

    Returns:
        Matplotlib axes object, or None if processing failed
    """
    # Extract RGB composite
    rgb_data = extract_rgb_composite(zip_file_path, bands=bands)
    if rgb_data is None:
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Display the satellite image
    bounds = rgb_data["bounds_wgs84"]
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])  # (min_lon, max_lon, min_lat, max_lat)

    ax.imshow(rgb_data["rgb_array"], extent=extent, aspect="auto")

    # Add target area overlay
    bbox_lons = [target_bbox[0], target_bbox[2], target_bbox[2], target_bbox[0], target_bbox[0]]
    bbox_lats = [target_bbox[1], target_bbox[1], target_bbox[3], target_bbox[3], target_bbox[1]]
    ax.plot(bbox_lons, bbox_lats, "red", linewidth=3, alpha=0.8, label="Target Area")

    # Zoom to target area with padding
    padding = 0.02
    ax.set_xlim(target_bbox[0] - padding, target_bbox[2] + padding)
    ax.set_ylim(target_bbox[1] - padding, target_bbox[3] + padding)

    # Customize plot
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)

    if title is None:
        title = f"Satellite Image\n{zip_file_path.name[:40]}..."
    ax.set_title(title, fontsize=11, fontweight="bold")

    ax.grid(True, alpha=0.3, color="white")
    ax.legend()

    return ax


def create_comparison_plot(
    zip_files: List[Path],
    target_bbox: List[float],
    center_lat: float,
    center_lon: float,
    figsize: Tuple[int, int] = (20, 8),
) -> plt.Figure:
    """Create a comparison plot with coverage map and satellite images.

    Args:
        zip_files: List of Sentinel-2 ZIP files
        target_bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84
        center_lat: Center latitude
        center_lon: Center longitude
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    num_images = min(len(zip_files), 2)  # Limit to 2 satellite images
    num_panels = num_images + 1  # Coverage map + satellite images

    fig, axes = plt.subplots(1, num_panels, figsize=figsize)

    # Ensure axes is always a list
    if num_panels == 1:
        axes = [axes]
    elif not hasattr(axes, "__len__"):
        axes = [axes]

    # Panel 1: Coverage map
    create_coverage_map(
        target_bbox, center_lat, center_lon, zip_files, ax=axes[0], title="Target Area Coverage"
    )

    # Panels 2+: Satellite images
    for idx, zip_file in enumerate(zip_files[:num_images]):
        ax_img = axes[idx + 1]

        result = display_satellite_image(
            zip_file,
            target_bbox,
            ax=ax_img,
            title=f"Satellite Image #{idx+1}\n{zip_file.name[:40]}...",
        )

        if result is None:
            # Show error message if processing failed
            ax_img.text(
                0.5,
                0.5,
                f"Error processing\n{zip_file.name}",
                ha="center",
                va="center",
                transform=ax_img.transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="lightcoral"),
            )
            ax_img.set_title("Processing Error", fontsize=12)
            ax_img.axis("off")

    plt.tight_layout()
    return fig


def create_metadata_summary(zip_files: List[Path], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Create a text summary of satellite data metadata.

    Args:
        zip_files: List of Sentinel-2 ZIP files
        ax: Matplotlib axes (creates new if None)

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.axis("off")

    if not zip_files:
        ax.text(
            0.5,
            0.5,
            "No satellite products found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("No Data Available", fontsize=14)
        return ax

    # Create metadata summary
    summary_text = "SATELLITE DATA SUMMARY\n" + "=" * 30 + "\n\n"

    for idx, zip_file in enumerate(zip_files[:3], 1):  # Show first 3
        size_mb = zip_file.stat().st_size / (1024 * 1024)

        summary_text += f"Product {idx}:\n"
        summary_text += f"  File: {zip_file.name[:50]}...\n"
        summary_text += f"  Size: {size_mb:.1f} MB\n"

        # Try to get image statistics
        rgb_data = extract_rgb_composite(zip_file)
        if rgb_data:
            stats = get_image_statistics(rgb_data)
            summary_text += f"  Resolution: {stats['shape'][0]}×{stats['shape'][1]} pixels\n"
            summary_text += f"  Coverage: {stats['coverage_area_km2']:.1f} km²\n"

        summary_text += "\n"

    if len(zip_files) > 3:
        summary_text += f"... and {len(zip_files) - 3} more products\n\n"

    summary_text += f"Total Products: {len(zip_files)}\n"
    total_size_gb = sum(f.stat().st_size for f in zip_files) / (1024**3)
    summary_text += f"Total Size: {total_size_gb:.2f} GB"

    # Display the text
    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=1", facecolor="white", alpha=0.9),
    )

    ax.set_title("Copernicus Data Summary", fontsize=14, fontweight="bold")

    return ax


def create_band_analysis_plot(
    zip_file_path: Path, bands_to_show: Optional[List[List[str]]] = None
) -> plt.Figure:
    """Create a multi-panel plot showing different band combinations.

    Args:
        zip_file_path: Path to Sentinel-2 ZIP file
        bands_to_show: List of band combinations to display

    Returns:
        Matplotlib figure object
    """
    if bands_to_show is None:
        bands_to_show = [
            ["B04", "B03", "B02"],  # Natural color (RGB)
            ["B08", "B04", "B03"],  # False color (NIR-R-G)
            ["B12", "B11", "B04"],  # SWIR composite
        ]

    fig, axes = plt.subplots(1, len(bands_to_show), figsize=(6 * len(bands_to_show), 6))

    if len(bands_to_show) == 1:
        axes = [axes]

    titles = ["Natural Color (RGB)", "False Color (NIR)", "SWIR Composite"]

    for idx, bands in enumerate(bands_to_show):
        rgb_data = extract_rgb_composite(zip_file_path, bands=bands)

        if rgb_data:
            bounds = rgb_data["bounds_wgs84"]
            extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

            axes[idx].imshow(rgb_data["rgb_array"], extent=extent, aspect="auto")
            axes[idx].set_title(f'{titles[idx]}\n{"-".join(bands)}', fontweight="bold")
            axes[idx].set_xlabel("Longitude (°E)")
            axes[idx].set_ylabel("Latitude (°N)")
        else:
            axes[idx].text(
                0.5,
                0.5,
                f'Error processing\n{"-".join(bands)}',
                ha="center",
                va="center",
                transform=axes[idx].transAxes,
            )
            axes[idx].set_title("Processing Error")

    plt.tight_layout()
    return fig
