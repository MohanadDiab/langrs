"""Geospatial coordinate conversion utilities."""

from typing import Optional
import rasterio


def get_crs(image_path: str) -> Optional[str]:
    """
    Extract CRS (Coordinate Reference System) from GeoTIFF using rasterio.
    
    Args:
        image_path: Path to GeoTIFF file
        
    Returns:
        CRS string (e.g., 'EPSG:32633') if available, None otherwise
        
    Example:
        >>> crs = get_crs("image.tif")
        >>> print(crs)
        'EPSG:32633'
    """
    try:
        with rasterio.open(image_path) as src:
            if src.crs:
                return src.crs.to_string()
    except Exception:
        # File might not be a GeoTIFF or might not have CRS
        pass
    return None
