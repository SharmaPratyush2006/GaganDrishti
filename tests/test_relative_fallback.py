"""The Phase 3 relative-field fallback: raw output of a FROZEN PRETRAINED model.

Skipped in full without torch.

No test here downloads anything. The published model is replaced by stand-ins
passed through ``build_pretrained_fallback(loader=...)``:

* a tiny hand-written module with the Hugging Face depth-model interface, for
  the wrapper's behaviour (freezing, raw passthrough, sizes, labelling);
* the real ``transformers`` Depth Anything class built from a tiny config with
  RANDOM weights, only to prove the wrapper speaks that class's actual API.

Neither is the pretrained model, and nothing here asserts or implies any
accuracy. The one test that fetches the real weights is opt-in via
``DEPTHWIZARD_ALLOW_DOWNLOAD=1``.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Phase 3 fallback requires PyTorch")

from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from depthwizard.relative.config import (  # noqa: E402
    InferenceConfig,
    ModelConfig,
    RelativeConfig,
    TrainingConfig,
)
from depthwizard.relative.fallback import (  # noqa: E402
    DEFAULT_PRETRAINED_FALLBACK,
    FALLBACK_WEIGHTS_SOURCE,
    PRETRAINED_FALLBACK_KIND,
    RANDOM_INIT_STUB_KIND,
    PretrainedRelativeFallback,
    assert_not_fallback,
    build_pretrained_fallback,
    fallback_kind,
    is_fallback,
    is_pretrained_fallback,
)
from depthwizard.relative.inference import infer_tile, save_prediction  # noqa: E402
from depthwizard.relative.model import PretrainedWeightsError  # noqa: E402


class FakeDepthModel(nn.Module):
    """The Hugging Face depth-estimation interface, with a known raw output."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.seen_sizes: list[tuple[int, int]] = []

    def forward(self, pixel_values):
        self.seen_sizes.append(tuple(pixel_values.shape[-2:]))
        return SimpleNamespace(predicted_depth=self.conv(pixel_values)[:, 0])


def _fallback(model: nn.Module | None = None, reason: str = "training stalled"):
    depth = model if model is not None else FakeDepthModel()
    return build_pretrained_fallback(reason=reason, loader=lambda _source: depth), depth


# ---------------------------------------------------------------------------
# Construction: pretrained or nothing
# ---------------------------------------------------------------------------


def test_default_source_is_depth_anything_v2_small():
    assert DEFAULT_PRETRAINED_FALLBACK == "depth-anything/Depth-Anything-V2-Small-hf"


def test_fallback_requires_a_reason():
    with pytest.raises(ValueError, match="non-empty reason"):
        build_pretrained_fallback(reason="  ", loader=lambda _s: FakeDepthModel())


def test_unobtainable_weights_raise_and_are_never_replaced_with_random():
    def offline(_source):
        raise OSError("no network")

    with pytest.raises(PretrainedWeightsError, match="NOT substitute random weights"):
        build_pretrained_fallback(reason="stalled", loader=offline)


def test_the_loader_is_asked_for_the_requested_source():
    asked = []
    build_pretrained_fallback(
        reason="stalled", source="some/model", loader=lambda s: asked.append(s) or FakeDepthModel()
    )
    assert asked == ["some/model"]


def test_a_pretrained_fallback_cannot_carry_the_random_init_label():
    with pytest.raises(ValueError, match="random_init"):
        PretrainedRelativeFallback(
            FakeDepthModel(), weights_source=FALLBACK_WEIGHTS_SOURCE, reason="x"
        )


# ---------------------------------------------------------------------------
# Frozen, and raw
# ---------------------------------------------------------------------------


def test_every_parameter_is_frozen_and_stays_in_eval():
    fallback, depth = _fallback()
    assert all(not p.requires_grad for p in fallback.parameters())
    fallback.train()
    assert fallback.training is False and depth.training is False
    assert fallback.describe()["params_trainable"] == 0


def test_no_graph_is_built():
    fallback, _ = _fallback()
    assert fallback(torch.randn(1, 3, 28, 28)).requires_grad is False


def test_output_is_the_raw_pretrained_prediction_with_no_rescaling():
    fallback, depth = _fallback()
    x = torch.randn(2, 3, 56, 56)  # a multiple of 14: no resampling at all
    with torch.no_grad():
        expected = depth.conv(x)
    torch.testing.assert_close(fallback(x), expected)


def test_512_tile_runs_at_504_and_returns_512():
    fallback, depth = _fallback()
    x = torch.randn(1, 3, 512, 512)
    out = fallback(x)
    assert depth.seen_sizes[-1] == (504, 504)  # nearest multiple of the 14 px patch
    assert out.shape == (1, 1, 512, 512)
    with torch.no_grad():
        raw = depth.conv(F.interpolate(x, size=(504, 504), mode="bilinear", align_corners=False))
    torch.testing.assert_close(
        out, F.interpolate(raw, size=(512, 512), mode="bilinear", align_corners=False)
    )


# ---------------------------------------------------------------------------
# Labelling: pretrained fallback vs random-init stub
# ---------------------------------------------------------------------------


def test_pretrained_fallback_is_labelled_pretrained_not_random():
    fallback, _ = _fallback()
    assert fallback.weights_source == f"hf:{DEFAULT_PRETRAINED_FALLBACK}"
    assert is_pretrained_fallback(fallback)
    assert not is_fallback(fallback)  # is_fallback means "random encoder"
    assert fallback_kind(fallback) == PRETRAINED_FALLBACK_KIND
    assert_not_fallback(fallback, context="report")  # pretrained: not refused

    description = fallback.describe()
    assert description["encoder_pretrained"] is True
    assert description["trained_on_dfc2019"] is False
    assert description["fallback_kind"] == PRETRAINED_FALLBACK_KIND
    assert description["fallback_reason"] == "training stalled"
    assert "NOT metres" in description["output_units"]
    assert "NOT metres" in description["output_quantity"]
    assert description["heads"] == ["relative_depth"]


def test_random_init_stub_is_never_labelled_a_pretrained_fallback():
    pytest.importorskip("timm", reason="builds the Phase 3 architecture")
    from depthwizard.relative.fallback import build_fallback_model

    stub = build_fallback_model(
        ModelConfig(encoder="convnext_tiny", pretrained=False, allow_random_init=True),
        reason="offline test",
        output_size=32,
    )
    assert not is_pretrained_fallback(stub) and not is_pretrained_fallback(stub.model)
    assert fallback_kind(stub.model) == RANDOM_INIT_STUB_KIND
    assert stub.describe()["fallback_kind"] == RANDOM_INIT_STUB_KIND
    assert stub.describe()["encoder_pretrained"] is False


def test_local_directory_source_is_recorded_as_local(tmp_path):
    fallback = build_pretrained_fallback(
        reason="offline", source=tmp_path, loader=lambda _s: FakeDepthModel()
    )
    assert fallback.weights_source == f"local:{tmp_path}"


# ---------------------------------------------------------------------------
# Through inference, at the shipped 512 px tile
# ---------------------------------------------------------------------------


@pytest.fixture
def fallback_cfg(dataset_config, tmp_path) -> RelativeConfig:
    return RelativeConfig(
        dataset=dataset_config,
        output_dir=tmp_path / "out",
        training=TrainingConfig(
            image_size=32, num_workers=0, persistent_workers=False, precision="fp32",
            device="cpu",
        ),
        inference=InferenceConfig(tile_size=512),
    )


def test_inference_output_is_labelled_as_the_pretrained_fallback(fallback_cfg, tmp_path):
    from conftest import write_synthetic_pair

    image_path, height_path = write_synthetic_pair(tmp_path / "big", "JAX_009_001", size=512)
    fallback, _ = _fallback()
    prediction = infer_tile(fallback, fallback_cfg, image_path, height_path=height_path,
                            device="cpu")

    assert prediction.relative_height.shape == (512, 512)
    assert prediction.fallback_kind == PRETRAINED_FALLBACK_KIND
    assert prediction.is_fallback is False  # not random
    assert prediction.log_space is False  # raw inverse depth, not log height

    record = prediction.to_json()
    assert record["is_fallback_model"] is True
    assert record["fallback_kind"] == PRETRAINED_FALLBACK_KIND
    assert "PRETRAINED FALLBACK" in record["fallback_note"]
    assert "RANDOM" not in record["fallback_note"]
    assert "NOT metres" in record["output_units"]

    written = save_prediction(prediction, tmp_path / "inf", "fallback")
    assert np.load(written["array"]).shape == (512, 512)
    assert json.loads(written["json"].read_text(encoding="utf-8"))["fallback_kind"] == (
        PRETRAINED_FALLBACK_KIND
    )


def test_trained_model_output_has_no_fallback_kind(fallback_cfg, tmp_path):
    from conftest import write_synthetic_pair

    class Trained(nn.Module):
        weights_source = "timm:test"

        def __init__(self):
            super().__init__()
            self.head = nn.Conv2d(3, 1, 1)

        def forward(self, x):
            return self.head(x)

        def describe(self):
            return {"encoder": "stub", "weights_source": self.weights_source}

    image_path, _ = write_synthetic_pair(tmp_path / "big", "JAX_009_001", size=512)
    prediction = infer_tile(Trained(), fallback_cfg, image_path, device="cpu")
    assert prediction.fallback_kind == ""
    assert prediction.to_json()["is_fallback_model"] is False


# ---------------------------------------------------------------------------
# The real transformers class (tiny config, RANDOM weights -- API check only)
# ---------------------------------------------------------------------------


def _tiny_depth_anything():
    transformers = pytest.importorskip("transformers")
    backbone = transformers.Dinov2Config(
        hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64,
        patch_size=14, image_size=56, reshape_hidden_states=False,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    config = transformers.DepthAnythingConfig(
        backbone_config=backbone, neck_hidden_sizes=[8, 16, 32, 32],
        fusion_hidden_size=16, head_hidden_size=8, reassemble_hidden_size=32,
    )
    return transformers.DepthAnythingForDepthEstimation(config)


def test_wrapper_speaks_the_real_depth_anything_api_at_512():
    fallback, _ = _fallback(_tiny_depth_anything())
    out = fallback(torch.randn(1, 3, 512, 512))
    assert out.shape == (1, 1, 512, 512)
    assert torch.isfinite(out).all()
    assert all(not p.requires_grad for p in fallback.parameters())


@pytest.mark.skipif(
    os.environ.get("DEPTHWIZARD_ALLOW_DOWNLOAD") != "1",
    reason="downloads the real Depth Anything V2-Small weights; set DEPTHWIZARD_ALLOW_DOWNLOAD=1",
)
def test_real_pretrained_weights_load_and_run_at_512():
    fallback = build_pretrained_fallback(reason="download check")
    assert fallback.weights_source == f"hf:{DEFAULT_PRETRAINED_FALLBACK}"
    out = fallback(torch.randn(1, 3, 512, 512))
    assert out.shape == (1, 1, 512, 512)
    assert torch.isfinite(out).all()
