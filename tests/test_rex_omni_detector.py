import numpy as np
import pytest


def test_prompt_to_categories_comma_split():
    from langrs.models.detection.rex_omni import _prompt_to_categories

    assert _prompt_to_categories("a,b, c") == ["a", "b", "c"]
    assert _prompt_to_categories("  ") == []


def test_rex_omni_detector_flattens_boxes():
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


def test_rex_omni_detector_rejects_empty_categories():
    from langrs.models.detection.rex_omni import RexOmniDetector
    from langrs.utils.exceptions import DetectionError

    det = RexOmniDetector(model_path="dummy", backend="transformers")
    det._rex = object()
    det._loaded = True

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(DetectionError):
        det.detect(img, "   ,   ")


def test_rex_omni_detector_requires_loaded_model():
    from langrs.models.detection.rex_omni import RexOmniDetector
    from langrs.utils.exceptions import DetectionError

    det = RexOmniDetector(model_path="dummy", backend="transformers")
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(DetectionError, match="Model not loaded"):
        det.detect(img, "building, road")


def test_rex_omni_detector_raises_on_empty_result_list():
    from langrs.models.detection.rex_omni import RexOmniDetector
    from langrs.utils.exceptions import DetectionError

    det = RexOmniDetector(model_path="dummy", backend="transformers")

    class DummyRex:
        def inference(self, images, task, categories):
            return []

    det._rex = DummyRex()
    det._loaded = True

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(DetectionError, match="returned no results"):
        det.detect(img, "building, road")


def test_rex_omni_detector_raises_on_unsuccessful_result():
    from langrs.models.detection.rex_omni import RexOmniDetector
    from langrs.utils.exceptions import DetectionError

    det = RexOmniDetector(model_path="dummy", backend="transformers")

    class DummyRex:
        def inference(self, images, task, categories):
            return [{"success": False, "error": "backend unavailable"}]

    det._rex = DummyRex()
    det._loaded = True

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(DetectionError, match="backend unavailable"):
        det.detect(img, "building, road")


def test_rex_omni_detector_load_weights_passes_transformers_defaults(monkeypatch):
    from langrs.models.detection.rex_omni import RexOmniDetector

    captured = {}

    class DummyWrapper:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("langrs.models.detection.rex_omni.RexOmniWrapper", DummyWrapper)

    det = RexOmniDetector(model_path="dummy", backend="transformers")
    det.load_weights()

    assert det.is_loaded is True
    assert captured["model_path"] == "dummy"
    assert captured["backend"] == "transformers"
    assert captured["max_tokens"] == 4096
    assert captured["temperature"] == 0.0
    assert captured["top_p"] == 0.05
    assert captured["top_k"] == 1
    assert captured["repetition_penalty"] == 1.05
    assert captured["attn_implementation"] == "flash_attention_2"
    assert captured["device_map"] == "auto"


def test_rex_omni_detector_load_weights_cpu_device_forces_cpu_device_map(monkeypatch):
    from langrs.models.detection.rex_omni import RexOmniDetector

    captured = {}

    class DummyWrapper:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("langrs.models.detection.rex_omni.RexOmniWrapper", DummyWrapper)

    det = RexOmniDetector(model_path="dummy", backend="transformers", device="cpu")
    det.load_weights()

    assert det.is_loaded is True
    assert captured["device_map"] == "cpu"


def test_rex_omni_detector_detect_wraps_unexpected_exception():
    from langrs.models.detection.rex_omni import RexOmniDetector
    from langrs.utils.exceptions import DetectionError

    det = RexOmniDetector(model_path="dummy", backend="transformers")

    class DummyRex:
        def inference(self, images, task, categories):
            raise RuntimeError("boom")

    det._rex = DummyRex()
    det._loaded = True

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(DetectionError, match="Detection failed in Rex-Omni adapter"):
        det.detect(img, "building, road")

