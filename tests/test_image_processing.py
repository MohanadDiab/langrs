import unittest
import os
import tempfile
import numpy as np
import torch
from PIL import Image
from unittest.mock import patch, MagicMock
from langrs_old.image_processing import (
    split_raster, slice_image_with_overlap, localize_bounding_boxes,
    load_image_as_array, array_to_pil_image, ImageProcessor
)
from langrs_old.config import LangRSConfig

class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = LangRSConfig()
        self.config.set("text_input", "test object")

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_slice_image_with_overlap(self):
        image = Image.new('RGB', (100, 100))
        chunks = slice_image_with_overlap(image, chunk_size=50, overlap=10)
        self.assertEqual(len(chunks), 9)  # 3x3 grid of chunks

    def test_localize_bounding_boxes(self):
        boxes = [(0, 0, 10, 10), (5, 5, 15, 15)]
        localized = localize_bounding_boxes(boxes, 100, 100)
        self.assertEqual(localized, [(100, 100, 110, 110), (105, 105, 115, 115)])

    def test_array_to_pil_image(self):
        array = np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)
        image = array_to_pil_image(array)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (100, 100))

    @patch('langrs.image_processing.LangSAMProcessor')
    def test_image_processor_process_image(self, mock_lang_sam):
        # Mock the LangSAMProcessor
        mock_processor = MagicMock()
        mock_processor.process_image.return_value = ([(0, 0, 10, 10)], torch.zeros((1, 100, 100)))
        mock_lang_sam.return_value = mock_processor

        # Create a test image
        test_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        Image.new('RGB', (100, 100)).save(test_image_path)

        processor = ImageProcessor(self.config)
        image, boxes, mask = processor.process_image(test_image_path)

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(len(boxes), 1)
        self.assertIsInstance(mask, np.ndarray)

    # Add more tests for other functions and methods as needed

if __name__ == '__main__':
    unittest.main()