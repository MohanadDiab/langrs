import unittest
import numpy as np
import os
import tempfile
from PIL import Image
from langrs_old.visualization import (
    visualize_raster, visualize_bounding_boxes, visualize_sam_results,
    plot_confusion_matrix, plot_precision_recall_curve
)

class TestVisualization(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_image = np.random.rand(100, 100, 3)
        self.test_mask = np.random.randint(0, 2, (100, 100))

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_visualize_raster(self):
        save_path = os.path.join(self.temp_dir, "raster.png")
        visualize_raster(self.test_image, save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_visualize_bounding_boxes(self):
        save_path = os.path.join(self.temp_dir, "bboxes.png")
        bboxes = [(10, 10, 20, 20), (30, 30, 40, 40)]
        visualize_bounding_boxes(self.test_image, bboxes, save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_visualize_sam_results(self):
        save_path = os.path.join(self.temp_dir, "sam_results.png")
        visualize_sam_results(self.test_image, self.test_mask, save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_confusion_matrix(self):
        save_path = os.path.join(self.temp_dir, "confusion_matrix.png")
        cm = np.array([[10, 2], [3, 15]])
        plot_confusion_matrix(cm, ['class1', 'class2'], save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_precision_recall_curve(self):
        save_path = os.path.join(self.temp_dir, "pr_curve.png")
        precision = np.array([0.1, 0.5, 0.9])
        recall = np.array([0.9, 0.5, 0.1])
        plot_precision_recall_curve(precision, recall, save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_visualize_raster_with_pil_image(self):
        save_path = os.path.join(self.temp_dir, "raster_pil.png")
        pil_image = Image.fromarray((self.test_image * 255).astype('uint8'))
        visualize_raster(pil_image, save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_visualize_bounding_boxes_with_pil_image(self):
        save_path = os.path.join(self.temp_dir, "bboxes_pil.png")
        pil_image = Image.fromarray((self.test_image * 255).astype('uint8'))
        bboxes = [(10, 10, 20, 20), (30, 30, 40, 40)]
        visualize_bounding_boxes(pil_image, bboxes, save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_visualize_sam_results_with_pil_image(self):
        save_path = os.path.join(self.temp_dir, "sam_results_pil.png")
        pil_image = Image.fromarray((self.test_image * 255).astype('uint8'))
        visualize_sam_results(pil_image, self.test_mask, save_path)
        self.assertTrue(os.path.exists(save_path))

if __name__ == '__main__':
    unittest.main()