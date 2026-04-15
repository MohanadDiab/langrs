import sys
import types

import pytest
import torch


def _install_fake_transformers(monkeypatch, qwen_cls):
    class DummyProcessor:
        def __init__(self):
            self.tokenizer = types.SimpleNamespace(padding_side=None)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    fake_transformers = types.SimpleNamespace(
        AutoProcessor=DummyProcessor,
        Qwen2_5_VLForConditionalGeneration=qwen_cls,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_transformers_wrapper_requires_cuda(monkeypatch):
    from langrs.rex_omni.wrapper import RexOmniWrapper

    class DummyQwen:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            return types.SimpleNamespace(device=torch.device("cpu"))

    _install_fake_transformers(monkeypatch, DummyQwen)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA-enabled PyTorch"):
        RexOmniWrapper(model_path="dummy", backend="transformers")


def test_transformers_wrapper_falls_back_to_eager_when_flash_attn_missing(monkeypatch):
    from langrs.rex_omni.wrapper import RexOmniWrapper

    calls = []

    class DummyQwen:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls.append(kwargs)
            if kwargs.get("attn_implementation") == "flash_attention_2":
                raise ImportError("FlashAttention2 has been toggled on, but it cannot be used.")
            return types.SimpleNamespace(device=torch.device("cuda"))

    _install_fake_transformers(monkeypatch, DummyQwen)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    wrapper = RexOmniWrapper(model_path="dummy", backend="transformers")

    assert wrapper.model_type == "transformers"
    assert len(calls) == 2
    assert calls[0]["attn_implementation"] == "flash_attention_2"
    assert calls[1]["attn_implementation"] == "eager"
