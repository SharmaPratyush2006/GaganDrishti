"""The frozen encoder, the DPT decoder and the single height head.

Skipped in full without torch and timm.

Every model here is built with ``allow_random_init=True``, which means the
encoder weights are RANDOM. That is deliberate: these tests check architecture
-- tensor shapes, what is frozen, how many heads there are, what the model says
about itself -- and never prediction quality, which random weights could not
support. No accuracy is asserted anywhere in this file.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="Phase 3 model requires PyTorch")
pytest.importorskip("timm", reason="Phase 3 model requires timm")

from depthwizard.relative.config import ModelConfig  # noqa: E402
from depthwizard.relative.fallback import (  # noqa: E402
    FALLBACK_WEIGHTS_SOURCE,
    assert_not_fallback,
    build_fallback_model,
    is_fallback,
)
from depthwizard.relative.model import (  # noqa: E402
    ENCODER_SPECS,
    DPTDecoder,
    RelativeHeightModel,
    build_model,
)


def _random_cfg(encoder: str = "convnext_tiny") -> ModelConfig:
    """A config that builds without touching the network.

    ConvNeXt-Tiny by default: it needs no ``img_size`` override and builds
    faster than the ViT, which keeps the shape tests quick.
    """
    return ModelConfig(encoder=encoder, pretrained=False, allow_random_init=True)


@pytest.fixture(scope="module")
def convnext_model() -> RelativeHeightModel:
    return build_model(_random_cfg("convnext_tiny"), output_size=64)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_forward_returns_one_channel_at_the_requested_size(convnext_model):
    out = convnext_model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_output_size_can_be_overridden_per_call(convnext_model):
    assert convnext_model(torch.randn(1, 3, 64, 64), output_size=32).shape == (1, 1, 32, 32)


def test_input_is_resized_to_the_encoder_requirement_not_the_target(convnext_model):
    """A tile that is not the encoder's input size still yields the tile's size."""
    out = convnext_model(torch.randn(1, 3, 96, 96), output_size=96)
    assert out.shape == (1, 1, 96, 96)


@pytest.mark.slow
def test_vit_encoder_also_produces_the_requested_size():
    model = build_model(_random_cfg("dinov2_small"), output_size=64)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_encoder_input_scales_with_the_tile():
    """Training sizes are unchanged; a 512 tile is NOT squeezed back to 256."""
    dinov2, convnext = ENCODER_SPECS["dinov2_small"], ENCODER_SPECS["convnext_tiny"]
    assert dinov2.encoder_size_for(256) == dinov2.input_size == 252
    assert convnext.encoder_size_for(256) == convnext.input_size == 256
    assert dinov2.encoder_size_for(512) == 504  # 36 patches of 14
    assert convnext.encoder_size_for(512) == 512


def _shapes_seen(model, tile):
    """Run a forward pass and record what the encoder and the head saw."""
    seen = {}
    hooks = [
        model.encoder.register_forward_hook(
            lambda _m, args, _out: seen.__setitem__("encoder_in", tuple(args[0].shape))
        ),
        model.head.register_forward_hook(
            lambda _m, args, _out: seen.__setitem__("head_in", tuple(args[0].shape))
        ),
    ]
    try:
        with torch.no_grad():
            seen["out"] = tuple(model(torch.randn(1, 3, tile, tile), output_size=tile).shape)
    finally:
        for hook in hooks:
            hook.remove()
    return seen


def test_512_inference_is_a_genuine_512_forward_pass():
    """Built (and trained) for 256, run at 512: every stage runs at 512 scale."""
    model = build_model(_random_cfg("convnext_tiny"), output_size=256).eval()
    at_256 = _shapes_seen(model, 256)
    at_512 = _shapes_seen(model, 512)

    assert at_256["encoder_in"][-2:] == (256, 256)
    assert at_512["encoder_in"][-2:] == (512, 512)
    # The decoder's resolution doubles with the tile: this is not a 256 px
    # prediction upsampled to 512.
    assert at_512["head_in"][-2:] == (2 * at_256["head_in"][-2], 2 * at_256["head_in"][-1])
    assert at_512["out"] == (1, 1, 512, 512)


@pytest.mark.slow
def test_vit_encoder_runs_natively_at_512():
    model = build_model(_random_cfg("dinov2_small"), output_size=256).eval()
    seen = _shapes_seen(model, 512)
    assert seen["encoder_in"][-2:] == (504, 504)
    assert seen["head_in"][-2:] == (288, 288)  # 36 patches -> 144 -> 288
    assert seen["out"] == (1, 1, 512, 512)
    # And still exactly the training geometry at 256.
    seen = _shapes_seen(model, 256)
    assert seen["encoder_in"][-2:] == (252, 252)
    assert seen["head_in"][-2:] == (144, 144)


def test_non_rgb_input_is_refused(convnext_model):
    with pytest.raises(ValueError, match=r"\[B, 3, H, W\]"):
        convnext_model(torch.randn(2, 1, 64, 64))


def test_unbatched_input_is_refused(convnext_model):
    with pytest.raises(ValueError, match=r"\[B, 3, H, W\]"):
        convnext_model(torch.randn(3, 64, 64))


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------


def test_every_encoder_parameter_is_frozen(convnext_model):
    assert all(not p.requires_grad for p in convnext_model.encoder.parameters())


def test_decoder_and_head_are_trainable(convnext_model):
    assert all(p.requires_grad for p in convnext_model.decoder.parameters())
    assert all(p.requires_grad for p in convnext_model.head.parameters())


def test_trainable_parameters_exclude_the_encoder(convnext_model):
    trainable = {id(p) for p in convnext_model.trainable_parameters()}
    encoder = {id(p) for p in convnext_model.encoder.parameters()}
    assert trainable and not (trainable & encoder)


def test_train_mode_keeps_the_backbone_in_eval(convnext_model):
    """Otherwise norm statistics would drift and 'frozen' would be a lie."""
    convnext_model.train()
    assert convnext_model.training is True
    assert convnext_model.encoder.backbone.training is False


def test_parameter_counts_add_up(convnext_model):
    counts = convnext_model.parameter_counts()
    assert counts["total"] == counts["trainable"] + counts["frozen"]
    assert counts["frozen"] > counts["trainable"]  # the backbone dominates


def test_no_gradient_reaches_the_encoder_after_a_backward(convnext_model):
    out = convnext_model(torch.randn(1, 3, 64, 64))
    out.mean().backward()
    assert all(p.grad is None for p in convnext_model.encoder.parameters())
    assert any(p.grad is not None for p in convnext_model.head.parameters())


# ---------------------------------------------------------------------------
# Scope: exactly one head, and it is not metres
# ---------------------------------------------------------------------------


def test_there_is_exactly_one_head(convnext_model):
    assert RelativeHeightModel.HEAD_NAMES == ("height",)
    assert convnext_model.describe()["output_channels"] == 1


def test_describe_states_the_output_is_not_metres(convnext_model):
    description = convnext_model.describe()
    assert "NOT metres" in description["output_units"]
    assert description["encoder_frozen"] is True


def test_describe_records_the_weight_source(convnext_model):
    assert convnext_model.describe()["weights_source"] == FALLBACK_WEIGHTS_SOURCE


# ---------------------------------------------------------------------------
# Encoder specs
# ---------------------------------------------------------------------------


def test_every_declared_encoder_has_a_spec():
    from depthwizard.relative.config import ENCODERS

    assert set(ENCODERS) == set(ENCODER_SPECS)


def test_specs_declare_four_pyramid_levels():
    for spec in ENCODER_SPECS.values():
        assert len(spec.feature_channels) == 4
        assert len(spec.feature_strides) == 4


def test_dinov2_input_size_is_a_multiple_of_its_patch_size():
    """252 = 18 patches of 14. A non-multiple would silently crop or pad."""
    assert ENCODER_SPECS["dinov2_small"].input_size % 14 == 0


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def test_decoder_requires_exactly_four_levels():
    with pytest.raises(ValueError, match="4 encoder levels"):
        DPTDecoder((96, 192, 384), decoder_channels=16)


def test_decoder_finishes_one_level_finer_than_its_finest_input():
    """Four fusion blocks, each doubling: 8 -> 16 with a finest input of 8."""
    decoder = DPTDecoder((8, 16, 32, 64), decoder_channels=8)
    levels = [
        torch.randn(1, 8, 8, 8),
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 32, 2, 2),
        torch.randn(1, 64, 1, 1),
    ]
    assert decoder(levels).shape == (1, 8, 16, 16)


# ---------------------------------------------------------------------------
# The fallback, which must never pass as a real model
# ---------------------------------------------------------------------------


def test_fallback_model_is_labelled_a_fallback():
    fallback = build_fallback_model(
        _random_cfg(), reason="no network access in the test environment", output_size=32
    )
    assert fallback.is_fallback
    assert is_fallback(fallback)
    assert is_fallback(fallback.model)


def test_fallback_describe_says_its_predictions_are_meaningless():
    fallback = build_fallback_model(_random_cfg(), reason="offline", output_size=32)
    description = fallback.describe()
    assert description["is_fallback"] is True
    assert description["encoder_pretrained"] is False
    assert description["predictions_are_meaningless"] is True
    assert description["fallback_reason"] == "offline"


def test_fallback_requires_a_reason():
    with pytest.raises(ValueError, match="non-empty reason"):
        build_fallback_model(_random_cfg(), reason="   ")


def test_fallback_does_not_mutate_the_callers_config():
    cfg = _random_cfg()
    build_fallback_model(cfg, reason="offline", output_size=32)
    assert cfg.allow_random_init is True  # unchanged, because it was already true
    pretrained_cfg = ModelConfig(encoder="convnext_tiny")
    build_fallback_model(pretrained_cfg, reason="offline", output_size=32)
    assert pretrained_cfg.pretrained is True  # the caller's object is untouched


def test_reporting_a_fallback_result_is_refused():
    fallback = build_fallback_model(_random_cfg(), reason="offline", output_size=32)
    with pytest.raises(RuntimeError, match="refusing to report validation metrics"):
        assert_not_fallback(fallback, context="report validation metrics")
