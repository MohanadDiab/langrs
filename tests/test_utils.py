import unittest
import os
import tempfile
import numpy as np
from PIL import Image
import json
from langrs_old.utils import (
    load_image_as_array, array_to_pil_image, sort_bboxes_by_area,
    save_json, load_json, ensure_dir, get_file_extension
)

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_load_image_as_array(self):
        # Create a test image
        image_path = os.path.join(self.temp_dir, "test_image.png")
        Image.new('RGB', (100, 100)).save(image_path)
        
        image_array = load_image_as_array(image_path)
        self.assertIsInstance(image_array, np.ndarray)
        self.assertEqual(image_array.shape, (100, 100, 3))

    def test_array_to_pil_image(self):
        array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil_image = array_to_pil_image(array)
        self.assertIsInstance(pil_image, Image.Image)
        self.assertEqual(pil_image.size, (100, 100))

    def test_sort_bboxes_by_area(self):
        bboxes = [(0, 0, 10, 10), (0, 0, 5, 5), (0, 0, 20, 20)]
        sorted_bboxes = sort_bboxes_by_area(bboxes)
        self.assertEqual(sorted_bboxes, [(0, 0, 20, 20), (0, 0, 10, 10), (0, 0, 5, 5)])

    def test_save_and_load_json(self):
        data = {"key": "value"}
        file_path = os.path.join(self.temp_dir, "test.json")
        save_json(data, file_path)
        loaded_data = load_json(file_path)
        self.assertEqual(data, loaded_data)

    def test_ensure_dir(self):
        new_dir = os.path.join(self.temp_dir, "new_directory")
        ensure_dir(new_dir)
        self.assertTrue(os.path.exists(new_dir))

    def test_get_file_extension(self):
        self.assertEqual(get_file_extension("image.jpg"), "jpg")
        self.assertEqual(get_file_extension("path/to/file.PNG"), "png")
        self.assertEqual(get_file_extension("no_extension"), "")

if __name__ == '__main__':
    unittest.main()