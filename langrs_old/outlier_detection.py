import numpy as np
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetector:
    """
    A class for detecting outliers using various methods.
    """

    def __init__(self, method: str = 'isolation_forest', **kwargs):
        """
        Initialize the OutlierDetector with a specified method.

        Args:
            method (str): The outlier detection method to use.
            **kwargs: Additional parameters for the chosen method.
        """
        self.method = method
        self.kwargs = kwargs
        self.detector = None

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the detector to the data and predict outliers.

        Args:
            data (np.ndarray): The input data, shape (n_samples, n_features).

        Returns:
            np.ndarray: An array of labels where -1 indicates outliers and 1 indicates inliers.

        Raises:
            ValueError: If an unknown method is specified.
        """
        try:
            if self.method == 'z_score':
                return self._z_score_outliers(data, self.kwargs.get('threshold', 3))
            elif self.method == 'iqr':
                return self._iqr_outliers(data)
            elif self.method == 'elliptic_envelope':
                self.detector = EllipticEnvelope(**self.kwargs)
            elif self.method == 'one_class_svm':
                self.detector = OneClassSVM(**self.kwargs)
            elif self.method == 'isolation_forest':
                self.detector = IsolationForest(**self.kwargs)
            elif self.method == 'local_outlier_factor':
                self.detector = LocalOutlierFactor(**self.kwargs)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            if self.detector:
                return self.detector.fit_predict(data)
        except Exception as e:
            logging.error(f"Error in outlier detection: {str(e)}")
            raise

    @staticmethod
    def _z_score_outliers(data: np.ndarray, threshold: float = 3) -> np.ndarray:
        """
        Detect outliers using the Z-score method.

        Args:
            data (np.ndarray): The input data.
            threshold (float): The Z-score threshold for outlier detection.

        Returns:
            np.ndarray: An array of labels where -1 indicates outliers and 1 indicates inliers.
        """
        z_scores = stats.zscore(data)
        return np.where(np.abs(z_scores) > threshold, -1, 1)

    @staticmethod
    def _iqr_outliers(data: np.ndarray) -> np.ndarray:
        """
        Detect outliers using the Interquartile Range (IQR) method.

        Args:
            data (np.ndarray): The input data.

        Returns:
            np.ndarray: An array of labels where -1 indicates outliers and 1 indicates inliers.
        """
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.where(np.any((data < lower_bound) | (data > upper_bound), axis=1), -1, 1)

def apply_outlier_detection(bboxes: List[Tuple[float, float, float, float]], 
                            method: str = 'isolation_forest', 
                            **kwargs) -> List[Tuple[float, float, float, float]]:
    """
    Apply outlier detection to refine bounding boxes.

    Args:
        bboxes (List[Tuple[float, float, float, float]]): List of bounding boxes.
        method (str): Outlier detection method to use.
        **kwargs: Additional parameters for the chosen method.

    Returns:
        List[Tuple[float, float, float, float]]: Refined list of bounding boxes with outliers removed.
    """
    try:
        # Calculate areas of bounding boxes
        areas = np.array([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]).reshape(-1, 1)

        # Perform outlier detection
        detector = OutlierDetector(method=method, **kwargs)
        labels = detector.fit_predict(areas)

        # Filter out the outliers
        refined_bboxes = [bbox for bbox, label in zip(bboxes, labels) if label == 1]

        logging.info(f"Applied {method} outlier detection. Original boxes: {len(bboxes)}, Refined boxes: {len(refined_bboxes)}")
        return refined_bboxes
    except Exception as e:
        logging.error(f"Error in applying outlier detection: {str(e)}")
        raise