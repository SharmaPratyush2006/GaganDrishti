"""Phase 3 inference: checkpoint round-trip, per-tile prediction, the figure.

Skipped in full without torch; the checkpoint round-trip also needs timm.

Every raster here is SYNTHETIC (see ``conftest.write_synthetic_pair``) and every
model has RANDOM weights. What is under test is plumbing -- the prediction is
the model's raw output with no rescaling, invalid ground truth is never shown as
a height, fallback models are labelled, checkpoints restore the exact model.
Nothing here asserts or implies an accuracy.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Phase 3 inference requires PyTorch")

from torch import nn  # noqa: E402

from depthwizard.relative.config import (  # noqa: E402
    CheckpointConfig,
    InferenceConfig,
    ModelConfig,
    RelativeConfig,
    TrainingConfig,
)
from depthwizard.relative.inference import (  # noqa: E402
    OUTPUT_UNITS,
    CheckpointError,
    infer_tile,
    load_model_from_checkpoint,
    predict_relative_height,
    save_prediction,
)

from conftest import DFC_HEIGHT_SUBDIR, DFC_IMAGE_SUBDIR  # noqa: E402


class StubModel(nn.Module):
    """1x1 conv with the describe()/weights_source interface inference reads."""

    def __init__(self, weights_source: str = "test_stub") -> None:
        super().__init__()
        self.head = nn.Conv2d(3, 1, 1)
        self.weights_source = weights_source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def describe(self) -> dict:
        return {"encoder": "stub", "weights_source": self.weights_source}


@pytest.fixture
def cfg(dataset_config, tmp_path) -> RelativeConfig:
    return RelativeConfig(
        dataset=dataset_config,
        output_dir=tmp_path / "out",
        model=ModelConfig(encoder="convnext_tiny", pretrained=False, allow_random_init=True),
        training=TrainingConfig(
            image_size=32,
            batch_size=2,
            gradient_accumulation_steps=1,
            epochs=1,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
            precision="fp32",
            device="cpu",
            use_scheduler=False,
            max_steps_per_epoch=1,
        ),
        checkpoint=CheckpointConfig(dir=tmp_path / "ckpt", include_encoder=True),
    )


def _paths(root, stem="JAX_004_001"):
    return (
        root / DFC_IMAGE_SUBDIR / f"{stem}_RGB.tif",
        root / DFC_HEIGHT_SUBDIR / f"{stem}_AGL.tif",
    )


# ---------------------------------------------------------------------------
# Per-tile prediction
# ---------------------------------------------------------------------------


def test_prediction_is_the_raw_model_output_with_no_rescaling():
    model = StubModel()
    image = np.random.default_rng(0).standard_normal((3, 16, 16)).astype(np.float32)
    expected = model(torch.from_numpy(image)[None])[0, 0].detach().numpy()
    np.testing.assert_allclose(predict_relative_height(model, image), expected, rtol=1e-6)


def test_prediction_refuses_a_non_rgb_tile():
    with pytest.raises(ValueError, match=r"\(3, H, W\)"):
        predict_relative_height(StubModel(), np.zeros((1, 8, 8), dtype=np.float32))


def test_infer_tile_masks_invalid_ground_truth_as_nan(cfg, dfc_like_root):
    image_path, height_path = _paths(dfc_like_root)
    prediction = infer_tile(StubModel(), cfg, image_path, height_path=height_path, device="cpu")

    assert prediction.relative_height.shape == (32, 32)
    assert prediction.image_rgb.shape == (32, 32, 3)
    # The fixture writes 8 nodata rows at the top of every height raster.
    assert np.isnan(prediction.ground_truth[:8]).all()
    assert not prediction.valid[:8].any()
    assert np.isfinite(prediction.ground_truth[8:]).all()
    assert np.isfinite(prediction.relative_height).all()


def test_infer_tile_works_without_ground_truth(cfg, dfc_like_root):
    image_path, _ = _paths(dfc_like_root)
    prediction = infer_tile(StubModel(), cfg, image_path, row_off=32, col_off=16, device="cpu")
    assert prediction.ground_truth is None and prediction.valid is None
    assert prediction.metadata["row_off"] == 32 and prediction.metadata["col_off"] == 16


def test_image_only_and_paired_reads_feed_the_model_the_same_pixels(cfg, dfc_like_root):
    image_path, height_path = _paths(dfc_like_root)
    paired = infer_tile(StubModel(), cfg, image_path, height_path=height_path, device="cpu")
    alone = infer_tile(StubModel(), cfg, image_path, device="cpu")
    np.testing.assert_array_equal(paired.image_rgb, alone.image_rgb)


def test_outputs_are_written_and_state_the_units(cfg, dfc_like_root, tmp_path):
    image_path, height_path = _paths(dfc_like_root)
    prediction = infer_tile(StubModel(), cfg, image_path, height_path=height_path, device="cpu")
    written = save_prediction(prediction, tmp_path / "inf", "tile")

    assert all(path.is_file() for path in written.values())
    np.testing.assert_array_equal(np.load(written["array"]), prediction.relative_height)
    record = json.loads(written["json"].read_text(encoding="utf-8"))
    assert record["output_units"] == OUTPUT_UNITS
    assert "NOT metres" in record["output_units"]
    assert record["is_fallback_model"] is False
    assert record["has_ground_truth"] is True


def test_a_random_encoder_is_labelled_in_every_output(cfg, dfc_like_root, tmp_path):
    image_path, _ = _paths(dfc_like_root)
    prediction = infer_tile(StubModel("random_init"), cfg, image_path, device="cpu")
    assert prediction.is_fallback is True
    record = prediction.to_json()
    assert record["is_fallback_model"] is True
    assert "RANDOM" in record["fallback_note"]
    save_prediction(prediction, tmp_path / "inf", "fallback")  # figure renders too


# ---------------------------------------------------------------------------
# 512 x 512 inference (training stays at the configured, smaller tile)
# ---------------------------------------------------------------------------


@pytest.fixture
def large_scene(tmp_path):
    """One synthetic 512 px pair: big enough for a full 512 tile, no padding."""
    from conftest import write_synthetic_pair

    root = tmp_path / "large"
    return write_synthetic_pair(root, "JAX_009_001", size=512, nodata_rows=16)


def test_inference_tile_defaults_to_the_configured_512(cfg, large_scene):
    cfg = replace(cfg, inference=InferenceConfig(tile_size=512))
    assert cfg.training.image_size == 32  # the model was trained on small tiles
    image_path, height_path = large_scene
    prediction = infer_tile(StubModel(), cfg, image_path, height_path=height_path, device="cpu")
    assert prediction.relative_height.shape == (512, 512)
    assert prediction.image_rgb.shape == (512, 512, 3)
    assert prediction.ground_truth.shape == (512, 512)
    assert prediction.metadata["tile_size"] == 512
    assert not prediction.valid[:16].any() and prediction.valid[16:].all()


def test_real_architecture_predicts_a_full_resolution_512_field(cfg, large_scene):
    """Real decoder/head, trained-tile default of 32 px: output is still 512."""
    pytest.importorskip("timm", reason="builds the real architecture")
    from depthwizard.relative.model import build_model

    model = build_model(cfg.model, output_size=cfg.training.image_size)
    image_path, _ = large_scene
    prediction = infer_tile(model, cfg, image_path, device="cpu", tile_size=512)
    assert prediction.relative_height.shape == (512, 512)
    assert np.isfinite(prediction.relative_height).all()
    # A per-call override, not the model's 32 px training default.
    assert model.output_size == 32


def test_prediction_refuses_a_model_that_returns_the_wrong_size():
    class Shrinks(StubModel):
        def forward(self, x, *, output_size=None):
            return self.head(x)[..., :8, :8]

    image = np.zeros((3, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="refusing to resample"):
        predict_relative_height(Shrinks(), image)


def test_inference_cli_writes_a_512_prediction(cfg, large_scene, tmp_path):
    pytest.importorskip("timm", reason="the CLI loads a real checkpoint")
    from depthwizard.relative.inference import main

    cfg = replace(cfg, inference=InferenceConfig(tile_size=512))
    _, path = _trained_checkpoint(cfg)
    image_path, height_path = large_scene
    out = tmp_path / "cli_out"
    assert main([
        "--checkpoint", str(path), "--image", str(image_path),
        "--height", str(height_path), "--device", "cpu", "--out", str(out),
    ]) == 0
    array = np.load(next(out.glob("*_relative_height.npy")))
    assert array.shape == (512, 512)
    record = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    assert "NOT metres" in record["output_units"]


# ---------------------------------------------------------------------------
# Checkpoint round-trip (needs a real architecture, hence timm)
# ---------------------------------------------------------------------------


def _trained_checkpoint(cfg):
    pytest.importorskip("timm", reason="checkpoint round-trip builds the real model")
    from depthwizard.relative.dataset import build_dataloaders
    from depthwizard.relative.model import build_model
    from depthwizard.relative.train import Trainer

    model = build_model(cfg.model, output_size=cfg.training.image_size)
    trainer = Trainer(cfg, model=model, loaders=build_dataloaders(cfg))
    result = trainer.train_epoch(0)
    path = trainer.save_checkpoint(cfg.checkpoint.dir / "ck.pt", result)
    return trainer.model, path


def test_checkpoint_restores_the_exact_model(cfg):
    trained, path = _trained_checkpoint(cfg)
    restored, restored_cfg = load_model_from_checkpoint(path)

    assert restored_cfg.to_dict() == cfg.to_dict()
    batch = torch.randn(1, 3, 32, 32)
    trained.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(batch), trained(batch))


def test_restored_random_encoder_is_still_labelled_random(cfg):
    from depthwizard.relative.fallback import is_fallback

    _, path = _trained_checkpoint(cfg)
    restored, _ = load_model_from_checkpoint(path)
    assert restored.weights_source == "random_init"
    assert is_fallback(restored)


def test_restored_encoder_is_still_frozen(cfg):
    _, path = _trained_checkpoint(cfg)
    restored, _ = load_model_from_checkpoint(path)
    restored.train()
    assert not any(p.requires_grad for p in restored.encoder.parameters())
    assert restored.encoder.backbone.training is False


def test_random_encoder_checkpoint_without_the_encoder_is_refused(cfg):
    cfg = replace(cfg, checkpoint=replace(cfg.checkpoint, include_encoder=False))
    _, path = _trained_checkpoint(cfg)
    with pytest.raises(CheckpointError, match="cannot be reconstructed"):
        load_model_from_checkpoint(path)


def test_missing_checkpoint_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_from_checkpoint(tmp_path / "nope.pt")
