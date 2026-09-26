"""The training loop: device/precision resolution, accumulation, checkpoints.

Skipped in full without torch. Needs no timm: the trainer is driven with a tiny
stand-in model, because what is under test is the *loop* -- how many optimiser
steps it takes, what it does with an unusable batch, what it writes to a
checkpoint, what it refuses to claim -- and none of that depends on the
backbone.

Nothing here asserts an accuracy. The losses that appear are from a
two-parameter toy model on synthetic rasters and describe nothing but the loop.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="Phase 3 trainer requires PyTorch")

from torch import nn  # noqa: E402

from depthwizard.relative.config import (  # noqa: E402
    CheckpointConfig,
    RelativeConfig,
    TrainingConfig,
)
from depthwizard.relative.dataset import build_dataloaders  # noqa: E402
from depthwizard.relative.train import (  # noqa: E402
    EpochResult,
    PrecisionError,
    Trainer,
    TrainingReport,
    resolve_device,
    resolve_precision,
    set_seed,
)


class TinyModel(nn.Module):
    """A stand-in with the interface :class:`Trainer` actually uses.

    One 1x1 convolution: enough to have trainable parameters, a gradient and a
    describable identity, and cheap enough that a full epoch is instant.
    """

    HEAD_NAMES = ("height",)

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, 1, 1)
        self.weights_source = "random_init_test_stub"

    def forward(self, x: torch.Tensor, *, output_size: int | None = None) -> torch.Tensor:
        return self.head(x)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def describe(self) -> dict:
        return {
            "encoder": "test_stub",
            "weights_source": self.weights_source,
            "params_trainable": sum(p.numel() for p in self.parameters()),
            "output_units": "relative (unitless) - NOT metres",
        }


@pytest.fixture
def cpu_config(dataset_config, tmp_path) -> RelativeConfig:
    return RelativeConfig(
        dataset=dataset_config,
        output_dir=tmp_path / "out",
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
            log_every=1000,
            use_scheduler=False,
        ),
        checkpoint=CheckpointConfig(dir=tmp_path / "ckpt", include_encoder=True),
    )


@pytest.fixture
def trainer(cpu_config) -> Trainer:
    return Trainer(cpu_config, model=TinyModel(), loaders=build_dataloaders(cpu_config))


# ---------------------------------------------------------------------------
# Device and precision
# ---------------------------------------------------------------------------


def test_cpu_is_resolved_directly():
    assert resolve_device("cpu").type == "cpu"


def test_auto_never_raises():
    assert resolve_device("auto").type in ("cpu", "cuda")


def test_requesting_cuda_without_cuda_is_an_error_not_a_downgrade():
    """A run configured for the GPU must not quietly take 40x longer on the CPU."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available, so the refusal path cannot be exercised")
    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        resolve_device("cuda")


def test_fp32_is_always_available():
    resolved = resolve_precision(
        TrainingConfig(precision="fp32", num_workers=0, persistent_workers=False),
        torch.device("cpu"),
    )
    assert resolved.effective == "fp32"
    assert resolved.autocast_dtype is None
    assert resolved.was_substituted is False


def test_fp16_on_cpu_is_refused_by_default():
    cfg = TrainingConfig(precision="fp16", num_workers=0, persistent_workers=False)
    with pytest.raises(PrecisionError, match="allow_precision_fallback"):
        resolve_precision(cfg, torch.device("cpu"))


def test_precision_fallback_is_recorded_not_hidden():
    cfg = TrainingConfig(
        precision="fp16",
        allow_precision_fallback=True,
        num_workers=0,
        persistent_workers=False,
    )
    resolved = resolve_precision(cfg, torch.device("cpu"))
    assert resolved.effective == "fp32"
    assert resolved.requested == "fp16"
    assert resolved.was_substituted is True
    assert resolved.substitution_reason
    assert resolved.to_dict()["was_substituted"] is True


def test_bf16_on_cpu_is_supported():
    cfg = TrainingConfig(precision="bf16", num_workers=0, persistent_workers=False)
    resolved = resolve_precision(cfg, torch.device("cpu"))
    assert resolved.effective == "bf16"
    assert resolved.autocast_dtype == torch.bfloat16


def test_set_seed_makes_initialisation_reproducible():
    set_seed(123)
    first = torch.randn(4)
    set_seed(123)
    assert torch.equal(first, torch.randn(4))


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_one_epoch_takes_gradient_steps_and_changes_the_weights(trainer):
    before = trainer.model.head.weight.detach().clone()
    result = trainer.train_epoch(0)
    assert result.optimizer_steps > 0
    assert result.micro_batches > 0
    assert math.isfinite(result.train_loss)
    assert not torch.equal(before, trainer.model.head.weight.detach())


def test_accumulation_reduces_the_optimiser_step_count(cpu_config):
    from dataclasses import replace

    single = Trainer(
        cpu_config, model=TinyModel(), loaders=build_dataloaders(cpu_config)
    ).train_epoch(0)

    accumulated_cfg = replace(
        cpu_config,
        training=replace(cpu_config.training, gradient_accumulation_steps=4),
    )
    accumulated = Trainer(
        accumulated_cfg,
        model=TinyModel(),
        loaders=build_dataloaders(accumulated_cfg),
    ).train_epoch(0)

    assert accumulated.micro_batches == single.micro_batches
    assert accumulated.optimizer_steps < single.optimizer_steps


def test_leftover_gradients_are_flushed_at_the_end_of_an_epoch(cpu_config):
    """A partial accumulation group still takes a step; it is real work."""
    from dataclasses import replace

    from depthwizard.relative.data import expected_steps

    cfg = replace(
        cpu_config, training=replace(cpu_config.training, gradient_accumulation_steps=4)
    )
    loaders = build_dataloaders(cfg)
    result = Trainer(cfg, model=TinyModel(), loaders=loaders).train_epoch(0)
    assert result.optimizer_steps == expected_steps(
        len(loaders[0].dataset), cfg.training.batch_size, 4
    )


def test_max_steps_per_epoch_caps_the_epoch(cpu_config):
    from dataclasses import replace

    cfg = replace(
        cpu_config, training=replace(cpu_config.training, max_steps_per_epoch=1)
    )
    result = Trainer(
        cfg, model=TinyModel(), loaders=build_dataloaders(cfg)
    ).train_epoch(0)
    assert result.optimizer_steps == 1


def test_set_epoch_is_propagated_to_the_dataset(trainer):
    trainer.train_epoch(3)
    assert trainer.train_loader.dataset.epoch == 3


def test_a_model_with_nothing_trainable_is_refused(cpu_config):
    model = TinyModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(RuntimeError, match="no trainable parameters"):
        Trainer(cpu_config, model=model, loaders=build_dataloaders(cpu_config))


# ---------------------------------------------------------------------------
# Validation honesty
# ---------------------------------------------------------------------------


def test_validation_produces_a_loss_and_metrics(trainer):
    loss, metrics = trainer.validate(0)
    assert loss is not None and math.isfinite(loss)
    assert metrics is not None and "si_rmse" in metrics


def test_no_validation_loader_means_not_measured_not_zero(cpu_config):
    loaders = build_dataloaders(cpu_config)
    trainer = Trainer(cpu_config, model=TinyModel(), loaders=(loaders[0], None))
    assert trainer.validate(0) == (None, None)


def test_validation_counts_are_summed_across_the_split(trainer):
    """valid_pixels is a total over every scored tile, not a per-batch average."""
    from depthwizard.relative.loss import scale_invariant_loss

    expected = 0.0
    batches = 0
    with torch.no_grad():
        for batch in trainer.val_loader:
            prediction = trainer.model(batch["image"])
            if scale_invariant_loss(
                prediction, batch["height"], batch["valid"], trainer.cfg.loss
            ).usable_count:
                expected += float(batch["valid"].sum())
                batches += 1
    assert batches > 1, "fixture must yield several scored batches to test pooling"
    _, metrics = trainer.validate(0)
    assert metrics["valid_pixels"] == pytest.approx(expected)
    assert metrics["scale_aligned"] == 1.0


def _limited(cpu_config, max_val_crops):
    from dataclasses import replace

    return replace(
        cpu_config, training=replace(cpu_config.training, max_val_crops=max_val_crops)
    )


def test_validation_scores_only_max_val_crops(cpu_config):
    from conftest import captured_depthwizard_logs

    full = Trainer(cpu_config, model=TinyModel(), loaders=build_dataloaders(cpu_config))
    full.validate(0)
    assert full.last_val_crops == 18  # 2 OMA pairs x 9 crops: the full grid

    cfg = _limited(cpu_config, 5)
    trainer = Trainer(cfg, model=TinyModel(), loaders=build_dataloaders(cfg))
    with captured_depthwizard_logs() as records:
        loss, _ = trainer.validate(0)
    assert loss is not None
    assert trainer.last_val_crops == 5
    logged = [r for r in records if r.getMessage() == "validation"]
    assert len(logged) == 1
    assert logged[0].val_crops_evaluated == 5
    assert logged[0].val_crops_available == 18
    assert logged[0].subset is True


def test_limited_validation_scores_the_same_crops_every_epoch(cpu_config):
    cfg = _limited(cpu_config, 5)
    trainer = Trainer(cfg, model=TinyModel(), loaders=build_dataloaders(cfg))
    first = [i for b in trainer.val_loader for i in (m["index"] for m in b["metadata"])]
    trainer.train_epoch(1)
    second = [i for b in trainer.val_loader for i in (m["index"] for m in b["metadata"])]
    assert first == second and len(first) == 5


def test_fit_records_validation_crops_in_the_report(cpu_config):
    cfg = _limited(cpu_config, 5)
    report = Trainer(cfg, model=TinyModel(), loaders=build_dataloaders(cfg)).fit()
    assert [e.val_crops for e in report.epochs] == [5]
    assert report.to_dict()["epochs"][0]["val_crops"] == 5
    assert report.dataset_summary["val_samples"] == 5
    assert report.dataset_summary["val_samples_available"] == 18
    assert report.dataset_summary["max_val_crops"] == 5
    assert "val crops/epoch   : 5 of 18  (fixed subset" in report.summary()


def test_fit_with_no_limit_reports_the_full_grid(cpu_config):
    cfg = _limited(cpu_config, None)
    report = Trainer(cfg, model=TinyModel(), loaders=build_dataloaders(cfg)).fit()
    assert report.epochs[0].val_crops == 18
    assert "val crops/epoch   : 18 of 18  (full grid)" in report.summary()


def test_no_validation_split_records_no_validation_crops(cpu_config):
    loaders = build_dataloaders(cpu_config)
    report = Trainer(cpu_config, model=TinyModel(), loaders=(loaders[0], None)).fit()
    assert report.epochs[0].val_crops is None
    assert "val crops/epoch" not in report.summary()


def test_metric_pooling_weights_each_metric_by_what_it_averaged():
    from depthwizard.relative.train import _MetricAggregate

    aggregate = _MetricAggregate()
    aggregate.add(
        {"si_rmse": 1.0, "abs_rel_aligned": 1.0, "valid_pixels": 100.0,
         "scored_samples": 1.0, "scale_aligned": 1.0}
    )
    aggregate.add(
        {"si_rmse": 4.0, "abs_rel_aligned": 4.0, "valid_pixels": 300.0,
         "scored_samples": 3.0, "scale_aligned": 1.0}
    )
    pooled = aggregate.result()
    assert pooled["si_rmse"] == pytest.approx((1.0 * 1 + 4.0 * 3) / 4)
    assert pooled["abs_rel_aligned"] == pytest.approx((100.0 + 1200.0) / 400)
    assert pooled["valid_pixels"] == 400.0
    assert pooled["scored_samples"] == 4.0


def test_metric_pooling_of_nothing_is_not_measured():
    from depthwizard.relative.train import _MetricAggregate

    aggregate = _MetricAggregate()
    aggregate.add(
        {"si_rmse": float("nan"), "abs_rel_aligned": float("nan"),
         "valid_pixels": 0.0, "scored_samples": 0.0, "scale_aligned": 1.0}
    )
    assert aggregate.result() is None


def test_report_renders_an_unmeasured_validation_honestly():
    report = TrainingReport(
        run_id="t",
        config={},
        model={"encoder": "stub", "weights_source": "x", "params_trainable": 1},
        device="cpu",
        precision={"effective": "fp32", "requested": "fp32", "was_substituted": False},
        epochs=[
            EpochResult(
                epoch=0,
                train_loss=0.5,
                optimizer_steps=1,
                micro_batches=1,
                skipped_batches=0,
                seconds=0.1,
                learning_rate=1e-4,
                val_loss=None,
            )
        ],
    )
    summary = report.summary()
    assert "not measured" in summary
    assert "0.00000" not in summary.split("val")[1].split("steps")[0]
    assert "NOT metres" in summary


def test_report_warns_loudly_about_a_fallback_model():
    report = TrainingReport(
        run_id="t",
        config={},
        model={"encoder": "stub", "weights_source": "random_init", "params_trainable": 1},
        device="cpu",
        precision={"effective": "fp32", "requested": "fp32", "was_substituted": False},
        used_fallback_model=True,
    )
    assert "RANDOM" in report.summary()


def test_report_dict_states_the_units():
    report = TrainingReport(
        run_id="t", config={}, model={}, device="cpu", precision={}
    )
    assert "NOT metres" in report.to_dict()["output_units"]


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def _result(epoch: int = 0) -> EpochResult:
    return EpochResult(
        epoch=epoch,
        train_loss=0.5,
        optimizer_steps=1,
        micro_batches=1,
        skipped_batches=0,
        seconds=0.1,
        learning_rate=1e-4,
    )


def test_checkpoint_embeds_the_config_and_the_units(trainer, tmp_path):
    path = trainer.save_checkpoint(tmp_path / "ck.pt", _result())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 0
    assert payload["config"]["run_id"] == "phase3"
    assert "NOT metres" in payload["output_units"]
    assert payload["precision"]["effective"] == "fp32"


def test_checkpoint_omits_the_frozen_encoder_by_default(cpu_config):
    from dataclasses import replace

    cfg = replace(
        cpu_config, checkpoint=replace(cpu_config.checkpoint, include_encoder=False)
    )
    trainer = Trainer(cfg, model=TinyModel(), loaders=build_dataloaders(cfg))
    state = trainer._state_to_save()
    assert not any(key.startswith("encoder.") for key in state)


def test_resume_restores_the_epoch_counter(cpu_config, tmp_path):
    from dataclasses import replace

    first = Trainer(cpu_config, model=TinyModel(), loaders=build_dataloaders(cpu_config))
    path = first.save_checkpoint(tmp_path / "resume.pt", _result(epoch=2))

    cfg = replace(
        cpu_config, checkpoint=replace(cpu_config.checkpoint, resume=str(path))
    )
    resumed = Trainer(cfg, model=TinyModel(), loaders=build_dataloaders(cfg))
    assert resumed.start_epoch == 3


def test_resume_restores_the_best_epoch(cpu_config, tmp_path):
    from dataclasses import replace

    first = Trainer(cpu_config, model=TinyModel(), loaders=build_dataloaders(cpu_config))
    first.best_val_loss, first.best_epoch = 0.25, 1
    path = first.save_checkpoint(tmp_path / "resume.pt", _result(epoch=2))

    cfg = replace(
        cpu_config, checkpoint=replace(cpu_config.checkpoint, resume=str(path))
    )
    resumed = Trainer(cfg, model=TinyModel(), loaders=build_dataloaders(cfg))
    assert (resumed.best_val_loss, resumed.best_epoch) == (0.25, 1)


def test_checkpoint_carries_optimizer_state_and_the_measured_epoch(trainer, tmp_path):
    result = trainer.train_epoch(0)
    payload = torch.load(
        trainer.save_checkpoint(tmp_path / "ck.pt", result),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["optimizer_state"]["state"], "optimiser moments were not saved"
    assert payload["epoch_result"]["train_loss"] == pytest.approx(result.train_loss)
    assert "model_state" in payload and "scheduler_state" in payload


def test_resume_from_a_missing_checkpoint_is_an_error(cpu_config, tmp_path):
    from dataclasses import replace

    cfg = replace(
        cpu_config,
        checkpoint=replace(cpu_config.checkpoint, resume=str(tmp_path / "nope.pt")),
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Trainer(cfg, model=TinyModel(), loaders=build_dataloaders(cfg))


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_fit_runs_an_epoch_and_reports_it(trainer):
    report = trainer.fit()
    assert len(report.epochs) == 1
    assert report.dataset_summary["spatially_separated"] is True
    assert report.dataset_summary["split_mode"] == "scene_prefix"
    assert report.best_val_loss is not None
    assert (trainer.cfg.checkpoint.dir / "epoch_000.pt").is_file()
    assert (trainer.cfg.checkpoint.dir / "best.pt").is_file()


def test_fit_summary_mentions_that_no_real_accuracy_is_claimed(trainer):
    assert "no real-world accuracy is claimed" in trainer.fit().summary()
