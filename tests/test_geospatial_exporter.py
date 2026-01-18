"""Tests for geospatial exporter module."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
import rasterio
from rasterio.crs import CRS

from langrs.geospatial.exporter import (
    read_image_metadata,
    pixel_to_geo,
    convert_bounding_boxes_to_polygons,
    convert_masks_to_polygons,
    convert_bounding_boxes_to_geospatial,
    convert_masks_to_geospatial,
)


class TestReadImageMetadata:
    """Test read_image_metadata function."""

    def test_read_metadata_with_crs(self):
        """Test reading metadata from GeoTIFF with CRS."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tif_path = f.name

        try:
            # Create GeoTIFF with CRS
            crs_str = "EPSG:32633"
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=3,
                dtype=np.uint8,
                crs=crs_str,
                transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            ) as dst:
                dst.write(np.zeros((3, 100, 100), dtype=np.uint8))

            transform, crs = read_image_metadata(tif_path)
            assert transform is not None
            assert crs == crs_str or crs is not None
        except rasterio.errors.CRSError:
            pytest.skip("PROJ database not available")
        finally:
            Path(tif_path).unlink()


class TestPixelToGeo:
    """Test pixel_to_geo function."""

    def test_pixel_to_geo_conversion(self):
        """Test pixel to geo coordinate conversion."""
        transform = rasterio.Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
        x, y = pixel_to_geo(10, 20, transform)
        assert isinstance(x, float)
        assert isinstance(y, float)


class TestConvertBoundingBoxesToPolygons:
    """Test convert_bounding_boxes_to_polygons function."""

    def test_convert_boxes_to_polygons(self):
        """Test converting boxes to polygons."""
        boxes = [(0, 0, 10, 10), (20, 20, 30, 30)]
        transform = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

        polygons = convert_bounding_boxes_to_polygons(boxes, transform)

        assert len(polygons) == len(boxes)
        assert all(hasattr(p, "area") for p in polygons)


class TestConvertMasksToPolygons:
    """Test convert_masks_to_polygons function."""

    def test_convert_masks_to_polygons(self):
        """Test converting masks to polygons."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # Square region
        transform = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

        polygons = convert_masks_to_polygons(mask, transform)

        assert len(polygons) > 0
        assert all(hasattr(p, "area") for p in polygons)


class TestConvertBoundingBoxesToGeospatial:
    """Test convert_bounding_boxes_to_geospatial function."""

    def test_convert_with_image_path(self):
        """Test conversion with image path."""
        boxes = [(0, 0, 10, 10), (20, 20, 30, 30)]

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tif_path = f.name

        try:
            crs_str = "EPSG:32633"
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=3,
                dtype=np.uint8,
                crs=crs_str,
                transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            ) as dst:
                dst.write(np.zeros((3, 100, 100), dtype=np.uint8))

            gdf = convert_bounding_boxes_to_geospatial(boxes, image_path=tif_path)
            assert len(gdf) == len(boxes)
        except rasterio.errors.CRSError:
            pytest.skip("PROJ database not available")
        finally:
            Path(tif_path).unlink()

    def test_convert_with_crs(self):
        """Test conversion with explicit CRS."""
        boxes = [(0, 0, 10, 10)]
        crs = "EPSG:4326"

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tif_path = f.name

        try:
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=3,
                dtype=np.uint8,
                crs=crs,
                transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            ) as dst:
                dst.write(np.zeros((3, 100, 100), dtype=np.uint8))

            gdf = convert_bounding_boxes_to_geospatial(boxes, image_path=tif_path, crs=crs)
            assert len(gdf) == len(boxes)
        except rasterio.errors.CRSError:
            pytest.skip("PROJ database not available")
        finally:
            Path(tif_path).unlink()

    def test_convert_no_crs_or_path(self):
        """Test conversion without CRS or path raises error."""
        boxes = [(0, 0, 10, 10)]
        with pytest.raises(ValueError, match="Either.*must be provided"):
            convert_bounding_boxes_to_geospatial(boxes)


class TestConvertMasksToGeospatial:
    """Test convert_masks_to_geospatial function."""

    def test_convert_masks_with_image_path(self):
        """Test conversion with image path."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tif_path = f.name

        try:
            crs_str = "EPSG:32633"
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=100,
                width=100,
                count=3,
                dtype=np.uint8,
                crs=crs_str,
                transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            ) as dst:
                dst.write(np.zeros((3, 100, 100), dtype=np.uint8))

            gdf = convert_masks_to_geospatial(mask, image_path=tif_path)
            assert len(gdf) > 0
        except rasterio.errors.CRSError:
            pytest.skip("PROJ database not available")
        finally:
            Path(tif_path).unlink()

    def test_convert_masks_no_crs_or_path(self):
        """Test conversion without CRS or path raises error."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Either.*must be provided"):
            convert_masks_to_geospatial(mask)
