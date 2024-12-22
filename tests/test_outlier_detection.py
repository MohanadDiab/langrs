import unittest
import numpy as np
from langrs_old.outlier_detection import OutlierDetector, apply_outlier_detection

class TestOutlierDetection(unittest.TestCase):

    def setUp(self):
        self.normal_data = np.random.normal(0, 1, (100, 1))
        self.outlier_data = np.concatenate([self.normal_data, np.array([[10], [-10]])])

    def test_z_score_outliers(self):
        detector = OutlierDetector(method='z_score')
        labels = detector.fit_predict(self.outlier_data)
        self.assertEqual(np.sum(labels == -1), 2)  # Should detect 2 outliers

    def test_iqr_outliers(self):
        detector = OutlierDetector(method='iqr')
        labels = detector.fit_predict(self.outlier_data)
        self.assertEqual(np.sum(labels == -1), 2)  # Should detect 2 outliers

    def test_isolation_forest(self):
        detector = OutlierDetector(method='isolation_forest', contamination=0.1)
        labels = detector.fit_predict(self.outlier_data)
        self.assertGreater(np.sum(labels == -1), 0)  # Should detect some outliers

    def test_apply_outlier_detection(self):
        bboxes = [(0, 0, 1, 1), (0, 0, 2, 2), (0, 0, 10, 10)]  # The last one is an outlier
        refined_bboxes = apply_outlier_detection(bboxes, method='isolation_forest', contamination=0.1)
        self.assertLess(len(refined_bboxes), len(bboxes))  # Should have removed at least one bbox

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            OutlierDetector(method='unknown_method')

if __name__ == '__main__':
    unittest.main()