"""Outlier detection methods for bounding box filtering."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from ..utils.types import BoundingBox


class OutlierDetector(ABC):
    """
    Abstract base class for outlier detection methods.
    
    All outlier detection methods must implement this interface
    to be used with the LangRS pipeline.
    """

    @abstractmethod
    def detect(
        self,
        areas: np.ndarray,
        bounding_boxes: List[BoundingBox],
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """
        Detect outliers in bounding box areas.
        
        Args:
            areas: Array of bounding box areas (N, 1) or (N,)
            bounding_boxes: List of bounding boxes corresponding to areas
            
        Returns:
            Tuple of (predictions, filtered_boxes)
            - predictions: Array of -1 for outliers, 1 for inliers
            - filtered_boxes: List of bounding boxes with outliers removed
        """
        pass


class ZScoreOutlierDetector(OutlierDetector):
    """Z-score based outlier detection."""

    def __init__(self, threshold: float = 3.0):
        """
        Initialize Z-score outlier detector.
        
        Args:
            threshold: Z-score threshold (default: 3.0)
        """
        self.threshold = threshold

    def detect(
        self,
        areas: np.ndarray,
        bounding_boxes: List[BoundingBox],
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """Detect outliers using Z-score method."""
        areas = areas.flatten() if areas.ndim > 1 else areas
        z_scores = stats.zscore(areas)
        outliers = np.abs(z_scores) > self.threshold

        predictions = np.ones_like(areas)
        predictions[outliers] = -1

        filtered_boxes = [
            bbox for bbox, is_inlier in zip(bounding_boxes, predictions == 1)
        ]

        return predictions, filtered_boxes


class IQROutlierDetector(OutlierDetector):
    """Interquartile Range (IQR) based outlier detection."""

    def __init__(self, multiplier: float = 1.5):
        """
        Initialize IQR outlier detector.
        
        Args:
            multiplier: IQR multiplier (default: 1.5)
        """
        self.multiplier = multiplier

    def detect(
        self,
        areas: np.ndarray,
        bounding_boxes: List[BoundingBox],
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """Detect outliers using IQR method."""
        areas = areas.flatten() if areas.ndim > 1 else areas
        Q1 = np.percentile(areas, 25)
        Q3 = np.percentile(areas, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.multiplier * IQR
        upper_bound = Q3 + self.multiplier * IQR

        outliers = (areas < lower_bound) | (areas > upper_bound)

        predictions = np.ones_like(areas)
        predictions[outliers] = -1

        filtered_boxes = [
            bbox for bbox, is_inlier in zip(bounding_boxes, predictions == 1)
        ]

        return predictions, filtered_boxes


class RobustCovarianceOutlierDetector(OutlierDetector):
    """Robust covariance (Elliptic Envelope) based outlier detection."""

    def __init__(self, contamination: float = 0.25):
        """
        Initialize robust covariance outlier detector.
        
        Args:
            contamination: Expected proportion of outliers (default: 0.25)
        """
        self.contamination = contamination

    def detect(
        self,
        areas: np.ndarray,
        bounding_boxes: List[BoundingBox],
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """Detect outliers using robust covariance method."""
        areas = areas.reshape(-1, 1) if areas.ndim == 1 else areas
        robust_cov = EllipticEnvelope(contamination=self.contamination)
        predictions = robust_cov.fit_predict(areas)

        filtered_boxes = [
            bbox for bbox, pred in zip(bounding_boxes, predictions) if pred == 1
        ]

        return predictions, filtered_boxes


class SVMOutlierDetector(OutlierDetector):
    """One-Class SVM based outlier detection."""

    def __init__(self, nu: float = 0.1, kernel: str = "rbf", gamma: float = 0.1):
        """
        Initialize SVM outlier detector.
        
        Args:
            nu: Upper bound on fraction of outliers (default: 0.1)
            kernel: Kernel type ('rbf' or 'linear', default: 'rbf')
            gamma: Kernel coefficient for 'rbf' (default: 0.1)
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

    def detect(
        self,
        areas: np.ndarray,
        bounding_boxes: List[BoundingBox],
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """Detect outliers using One-Class SVM method."""
        areas = areas.reshape(-1, 1) if areas.ndim == 1 else areas
        svm = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        predictions = svm.fit_predict(areas)

        filtered_boxes = [
            bbox for bbox, pred in zip(bounding_boxes, predictions) if pred == 1
        ]

        return predictions, filtered_boxes


class IsolationForestOutlierDetector(OutlierDetector):
    """Isolation Forest based outlier detection."""

    def __init__(self, contamination: float = 0.25, random_state: int = 42):
        """
        Initialize Isolation Forest outlier detector.
        
        Args:
            contamination: Expected proportion of outliers (default: 0.25)
            random_state: Random state for reproducibility (default: 42)
        """
        self.contamination = contamination
        self.random_state = random_state

    def detect(
        self,
        areas: np.ndarray,
        bounding_boxes: List[BoundingBox],
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """Detect outliers using Isolation Forest method."""
        areas = areas.reshape(-1, 1) if areas.ndim == 1 else areas
        isolation_forest = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        )
        predictions = isolation_forest.fit_predict(areas)

        filtered_boxes = [
            bbox for bbox, pred in zip(bounding_boxes, predictions) if pred == 1
        ]

        return predictions, filtered_boxes


class LOFOutlierDetector(OutlierDetector):
    """Local Outlier Factor (LOF) based outlier detection."""

    def __init__(
        self, contamination: float = 0.25, n_neighbors: int = 20
    ):
        """
        Initialize LOF outlier detector.
        
        Args:
            contamination: Expected proportion of outliers (default: 0.25)
            n_neighbors: Number of neighbors (default: 20)
        """
        self.contamination = contamination
        self.n_neighbors = n_neighbors

    def detect(
        self,
        areas: np.ndarray,
        bounding_boxes: List[BoundingBox],
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """Detect outliers using LOF method."""
        areas = areas.reshape(-1, 1) if areas.ndim == 1 else areas
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, contamination=self.contamination
        )
        predictions = lof.fit_predict(areas)

        filtered_boxes = [
            bbox for bbox, pred in zip(bounding_boxes, predictions) if pred == 1
        ]

        return predictions, filtered_boxes
