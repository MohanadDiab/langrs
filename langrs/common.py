import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, shape
from rasterio.features import shapes
import rasterio
from samgeo.common import get_crs
from PIL import Image
from torchvision.ops import nms
import os


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


def load_image(image):
    """
    Load an image from file path, numpy array, or PIL Image object.

    Args:
        image (str | np.ndarray | PIL.Image.Image): Input image provided as:
                                                    - A file path (GeoTIFF, PNG, JPG)
                                                    - A NumPy array with shape (H, W, C)
                                                    - A PIL Image object

    Returns:
        tuple:
            - image_path (str or None): Original file path, or None if not from file.
            - pil_image (PIL.Image.Image): Loaded image as PIL object.
            - np_image (np.ndarray): Loaded image as NumPy array (H, W, C).
            - source_crs (str or None): Coordinate reference system if available (for GeoTIFF).

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        ValueError: If the image format is unsupported.
        TypeError: If the input type is not recognized.
    """

    if isinstance(image, str):
        if not os.path.isfile(image):
            raise FileNotFoundError(f"Image file not found: {image}")

        ext = os.path.splitext(image)[-1].lower()

        if ext in ['.tif', '.tiff']:
            with rasterio.open(image) as src:
                rgb_image = np.array(src.read([1, 2, 3]))
            pil_image = Image.fromarray(np.transpose(rgb_image, (1, 2, 0)))
            np_image = np.array(pil_image)
            source_crs = get_crs(image)

        elif ext in ['.jpg', '.jpeg', '.png']:
            pil_image = Image.open(image).convert('RGB')
            np_image = np.array(pil_image)
            source_crs = None  # No CRS for non-georeferenced images

        else:
            raise ValueError(f"Unsupported image format: {ext}")

        return image, pil_image, np_image, source_crs

    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[-1] not in [3, 4]:
            raise ValueError("Expected RGB(A) image with shape (H, W, 3) or (H, W, 4)")
        np_image = image[..., :3]  # Drop alpha if present
        pil_image = Image.fromarray(np_image.astype(np.uint8))
        return None, pil_image, np_image, None

    elif isinstance(image, Image.Image):
        pil_image = image.convert('RGB')
        np_image = np.array(pil_image)
        return None, pil_image, np_image, None

    else:
        raise TypeError("Unsupported image input. Must be file path (str), numpy array, or PIL image.")
        
def apply_nms(boxes, iou_threshold=0.5):
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.ones(len(boxes))  # Or real scores if you have them
    indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    return boxes_tensor[indices]