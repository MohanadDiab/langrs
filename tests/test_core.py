import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import json
import tempfile
import os
from PIL import Image
from langrs_old.core import LangRS

class TestLangRS(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.gt_bb_path = os.path.join(self.temp_dir, "gt_bb.json")
        self.gt_mask_path = os.path.join(self.temp_dir, "gt_mask.json")
        
        # Create mock config file
        config = {
            "image_input": "test.jpg",
            "outlier_methods": ["isolation_forest"],
            "output_dir": self.temp_dir,
            "evaluation": True,
            "ground_truth_bb": self.gt_bb_path,
            "ground_truth_mask": self.gt_mask_path
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
        
        # Create mock ground truth files
        with open(self.gt_bb_path, 'w') as f:
            json.dump([(0, 0, 10, 10), (20, 20, 30, 30)], f)
        with open(self.gt_mask_path, 'w') as f:
            json.dump(np.zeros((100, 100)).tolist(), f)

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    @patch('langrs.core.ImageProcessor')
    @patch('langrs.core.apply_outlier_detection')
    @patch('langrs.core.visualize_bounding_boxes')
    @patch('langrs.core.visualize_sam_results')
    @patch('langrs.core.evaluate_detection')
    @patch('langrs.core.evaluate_segmentation')
    def test_process_with_evaluation(self, mock_eval_seg, mock_eval_det, mock_vis_sam, mock_vis_bbox, mock_outlier, mock_img_proc):
        # Mock image processing results
        mock_image = MagicMock(spec=Image.Image)
        mock_bboxes = [(0, 0, 10, 10), (20, 20, 30, 30)]
        mock_mask = np.zeros((100, 100))
        mock_img_proc.return_value.process_image.return_value = (mock_image, mock_bboxes, mock_mask)

        # Mock outlier detection
        mock_outlier.return_value = mock_bboxes

        # Mock evaluation results
        mock_eval_det.return_value = {"Precision": 0.9, "Recall": 0.9}
        mock_eval_seg.return_value = {"Accuracy": 0.95, "F1 Score": 0.92}

        # Create LangRS instance and run processing
        lang_rs = LangRS(self.config_path)
        lang_rs.process()

        # Assertions
        mock_img_proc.return_value.process_image.assert_called_once()
        mock_outlier.assert_called_once()
        mock_vis_bbox.assert_called_once()
        mock_vis_sam.assert_called_once()
        mock_eval_det.assert_called_once()
        mock_eval_seg.assert_called_once()

        # Check if evaluation results were saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "evaluation_results.json")))

if __name__ == '__main__':
    unittest.main()