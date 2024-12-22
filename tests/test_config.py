import unittest
import json
import tempfile
import os
from langrs_old.config import LangRSConfig

class TestLangRSConfig(unittest.TestCase):

    def setUp(self):
        self.config = LangRSConfig()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_default_config(self):
        self.assertEqual(self.config.get("tile_size"), 1000)
        self.assertEqual(self.config.get("overlap"), 300)
        self.assertFalse(self.config.get("tiling"))

    def test_load_config(self):
        config_path = os.path.join(self.temp_dir, "test_config.json")
        test_config = {
            "text_input": "test input",
            "image_input": "test.tif",
            "tile_size": 500,
            "tiling": True
        }
        with open(config_path, 'w') as f:
            json.dump(test_config, f)

        self.config.load_config(config_path)
        self.assertEqual(self.config.get("text_input"), "test input")
        self.assertEqual(self.config.get("tile_size"), 500)
        self.assertTrue(self.config.get("tiling"))

    def test_save_config(self):
        self.config.set("text_input", "save test")
        config_path = os.path.join(self.temp_dir, "saved_config.json")
        self.config.save_config(config_path)

        with open(config_path, 'r') as f:
            loaded_config = json.load(f)

        self.assertEqual(loaded_config["text_input"], "save test")

    def test_validate_config(self):
        self.config.set("tile_size", -100)
        with self.assertRaises(ValueError):
            self.config.validate()

        self.config.set("tile_size", 1000)
        self.config.set("evaluation", True)
        with self.assertRaises(ValueError):
            self.config.validate()

    def test_str_representation(self):
        config_str = str(self.config)
        self.assertIn("tile_size", config_str)
        self.assertIn("overlap", config_str)

if __name__ == '__main__':
    unittest.main()