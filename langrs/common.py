import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, shape
from rasterio.features import shapes


def read_image_metadata(image_path):
    """Reads geotransform and CRS from GeoTIFF."""
    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs.to_string()
    return transform, crs


def pixel_to_geo(col, row, transform):
    """Converts pixel coordinates (col, row) to georeferenced coordinates (x, y)."""
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y


def convert_bounding_boxes_to_polygons(bounding_boxes, transform):
    """Converts bounding boxes (pixel coords) into georeferenced polygons."""
    polygons = []
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = [float(coord) for coord in box]

        # Convert all corners to geospatial coordinates
        top_left = pixel_to_geo(xmin, ymin, transform)
        top_right = pixel_to_geo(xmax, ymin, transform)
        bottom_right = pixel_to_geo(xmax, ymax, transform)
        bottom_left = pixel_to_geo(xmin, ymax, transform)

        # Create a rectangle polygon from 4 corners
        polygon = Polygon([top_left, top_right, bottom_right, bottom_left, top_left])
        polygons.append(polygon)

    return polygons


def convert_masks_to_polygons(masks, transform):
    """Converts binary masks into georeferenced polygons."""
    shapes_generator = shapes(masks.astype(np.uint8), mask=masks > 0, transform=transform)
    polygons = [shape(geom) for geom, value in shapes_generator if value > 0]
    return polygons


def convert_bounding_boxes_to_geospatial(bounding_boxes, image_path=None, crs=None):
    """
    Converts bounding boxes to a georeferenced GeoDataFrame.

    Parameters
    ----------
    bounding_boxes : list of tuples
        List of (xmin, ymin, xmax, ymax) in pixel coordinates.
    image_path : str, optional
        Path to source GeoTIFF to extract CRS/transform.
    crs : str, optional
        CRS string like "EPSG:32633". If None, read from image_path.

    Returns
    -------
    gdf_boxes : GeoDataFrame of georeferenced polygons (rectangles).
    """
    if image_path:
        transform, image_crs = read_image_metadata(image_path)
        if crs is None:
            crs = image_crs
    elif crs is None:
        raise ValueError("Either `crs` or `image_path` must be provided.")

    box_polygons = convert_bounding_boxes_to_polygons(bounding_boxes, transform)
    gdf_boxes = gpd.GeoDataFrame(geometry=gpd.GeoSeries(box_polygons), crs=crs)

    return gdf_boxes


def convert_masks_to_geospatial(masks, image_path=None, crs=None):
    """
    Converts segmentation mask to a georeferenced GeoDataFrame.

    Parameters
    ----------
    masks : np.ndarray
        Binary mask array (H, W) matching the image dimensions.
    image_path : str, optional
        Path to source GeoTIFF to extract CRS/transform.
    crs : str, optional
        CRS string like "EPSG:32633". If None, read from image_path.

    Returns
    -------
    gdf_masks : GeoDataFrame of georeferenced mask polygons.
    """
    if image_path:
        transform, image_crs = read_image_metadata(image_path)
        if crs is None:
            crs = image_crs
    elif crs is None:
        raise ValueError("Either `crs` or `image_path` must be provided.")

    mask_polygons = convert_masks_to_polygons(masks, transform)
    gdf_masks = gpd.GeoDataFrame(geometry=gpd.GeoSeries(mask_polygons), crs=crs)

    return gdf_masks
