import numpy as np
import pytest


def test_prompt_to_categories_comma_split():
    from langrs.models.detection.rex_omni import _prompt_to_categories

    assert _prompt_to_categories("a,b, c") == ["a", "b", "c"]
    assert _prompt_to_categories("  ") == []


def test_rex_omni_detector_flattens_boxes(monkeypatch):
    from langrs.models.detection.rex_omni import RexOmniDetector

    det = RexOmniDetector(model_path="dummy", backend="transformers")

    class DummyRex:
        def inference(self, images, task, categories):
            assert task == "detection"
            assert categories == ["building", "road"]
            return [
                {
                    "success": True,
                    "extracted_predictions": {
                        "building": [{"type": "box", "coords": [1, 2, 3, 4]}],
                        "road": [{"type": "point", "coords": [5, 6]}],
                    },
                }
            ]

    det._rex = DummyRex()
    det._loaded = True

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    boxes = det.detect(img, "building, road")
    assert boxes == [(1.0, 2.0, 3.0, 4.0)]


def test_rex_omni_detector_rejects_empty_categories(monkeypatch):
    from langrs.models.detection.rex_omni import RexOmniDetector
    from langrs.utils.exceptions import DetectionError

    det = RexOmniDetector(model_path="dummy", backend="transformers")
    det._rex = object()
    det._loaded = True

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(DetectionError):
        det.detect(img, "   ,   ")

