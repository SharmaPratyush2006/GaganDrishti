"""The Phase 3 training loop.

Sized for one 6 GB laptop GPU, and built so that every number it reports can be
traced to something it actually measured.

What the loop does, in order, per epoch::

    dataset.set_epoch(e)        # new crops, from a seed derived from e
    for micro_batch in loader:
        forward under autocast
        loss = scale_invariant_loss(...)          # masked, per sample
        (loss / accumulation_steps).backward()
        every `accumulation_steps` micro-batches:
            clip gradients, optimiser step, zero grads
    flush any leftover accumulated gradients
    validate on the held-out scenes (if there are any)
    checkpoint

Honesty rules this module enforces
----------------------------------
* **No fabricated metrics.** Validation numbers exist only when a validation
  split exists and produced at least one scored pixel. Otherwise the report
  carries ``None``, and :meth:`TrainingReport.summary` renders it as
  ``"not measured"`` -- never 0.0, never a placeholder.
* **No silent precision substitution.** If the requested precision is not
  supported, training either refuses (the default) or falls back to fp32 *and
  records the substitution* in the report, the logs and every checkpoint.
* **No unlabelled fallback model.** If the encoder had to be randomly
  initialised, that fact propagates into the report and the checkpoint via
  :mod:`depthwizard.relative.fallback`.
* **Relative, not metric.** Nothing here converts a prediction to metres. The
  model's output is unitless and the report says so on every line that mentions
  it.

No dataset ships with this repository. Running this module without a real
DFC2019 download on disk raises
:class:`~depthwizard.relative.data.DatasetError` naming the missing path, and
that is the intended behaviour -- it does not fall back to synthetic data.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from depthwizard.logging_setup import get_logger
from depthwizard.relative.config import RelativeConfig, TrainingConfig, load_relative_config
from depthwizard.relative.dataset import build_dataloaders, full_length
from depthwizard.relative.loss import LossOutput, relative_metrics, scale_invariant_loss
from depthwizard.relative.model import RelativeHeightModel, build_model

__all__ = [
    "PrecisionError",
    "ResolvedPrecision",
    "EpochResult",
    "TrainingReport",
    "Trainer",
    "resolve_device",
    "resolve_precision",
    "set_seed",
    "main",
]

log = get_logger(__name__)


class PrecisionError(RuntimeError):
    """Raised when the requested precision is unsupported and no fallback was allowed."""


# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch. Does not force deterministic kernels.

    Bit-exact reproducibility would need ``torch.use_deterministic_algorithms``,
    which makes several convolution backward kernels much slower and raises on
    others. Seeding gives run-to-run reproducibility of the data pipeline and
    the initialisation, which is what matters for comparing configurations; the
    remaining nondeterminism is cuDNN kernel selection, and it is noted here so
    nobody reads two slightly different losses as a bug.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Turn ``auto`` / ``cuda`` / ``cpu`` into a concrete device.

    ``cuda`` when CUDA is unavailable is an error, not a silent downgrade: a run
    that was configured for the GPU and quietly took 40x longer on the CPU is a
    worse outcome than a clear failure.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "training.device is 'cuda' but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch, or set training.device to 'auto' "
                "(which picks the CPU) or 'cpu' (which says so explicitly)."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ResolvedPrecision:
    """What precision training will actually run in, and whether that was asked for."""

    requested: str
    effective: str
    autocast_dtype: torch.dtype | None
    use_grad_scaler: bool
    #: Empty when the request was honoured; otherwise why it was not.
    substitution_reason: str = ""

    @property
    def was_substituted(self) -> bool:
        return self.effective != self.requested

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "was_substituted": self.was_substituted,
            "substitution_reason": self.substitution_reason,
        }


def resolve_precision(training: TrainingConfig, device: torch.device) -> ResolvedPrecision:
    """Decide the run's precision, refusing or substituting per the config.

    * ``fp32`` is always available.
    * ``bf16`` needs a CUDA device reporting bf16 support, or a CPU (where
      ``torch.autocast('cpu')`` supports bfloat16).
    * ``fp16`` is CUDA-only here, and brings a ``GradScaler``. CPU fp16 autocast
      exists but is slow and numerically fragile, so it is treated as
      unsupported rather than offered as a trap.
    """
    requested = training.precision
    if requested == "fp32":
        return ResolvedPrecision(requested, "fp32", None, False)

    unsupported = ""
    if requested == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            unsupported = (
                f"CUDA device {torch.cuda.get_device_name(device)} does not support bf16"
            )
    elif requested == "fp16":
        if device.type != "cuda":
            unsupported = "fp16 autocast requires a CUDA device"

    if not unsupported:
        dtype = torch.bfloat16 if requested == "bf16" else torch.float16
        return ResolvedPrecision(
            requested=requested,
            effective=requested,
            autocast_dtype=dtype,
            use_grad_scaler=(requested == "fp16" and device.type == "cuda"),
        )

    if not training.allow_precision_fallback:
        raise PrecisionError(
            f"training.precision is {requested!r} but {unsupported}. "
            "Set training.precision to a supported value, or set "
            "training.allow_precision_fallback: true to run in fp32 instead -- "
            "the substitution will then be recorded everywhere the run is reported."
        )

    log.warning(
        "requested precision is unsupported; falling back to fp32. This is "
        "recorded in the training report and in every checkpoint.",
        extra={"requested": requested, "reason": unsupported},
    )
    return ResolvedPrecision(
        requested=requested,
        effective="fp32",
        autocast_dtype=None,
        use_grad_scaler=False,
        substitution_reason=unsupported,
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class EpochResult:
    """Everything measured during one epoch. ``None`` means "not measured"."""

    epoch: int
    train_loss: float
    optimizer_steps: int
    micro_batches: int
    #: Micro-batches skipped because no sample in them had enough valid pixels.
    skipped_batches: int
    seconds: float
    learning_rate: float
    #: None when there is no validation split, or it scored no pixels.
    val_loss: float | None = None
    val_metrics: dict[str, float] | None = None
    #: Validation crops actually run through the model this epoch (None when
    #: there is no validation split). Bounded by ``training.max_val_crops``.
    val_crops: int | None = None
    peak_vram_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingReport:
    """The record of a run: its config, its environment, and what it measured."""

    run_id: str
    config: dict[str, Any]
    model: dict[str, Any]
    device: str
    precision: dict[str, Any]
    epochs: list[EpochResult] = field(default_factory=list)
    best_val_loss: float | None = None
    best_epoch: int | None = None
    #: True when the encoder's weights were random rather than pretrained.
    used_fallback_model: bool = False
    dataset_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config,
            "model": self.model,
            "device": self.device,
            "precision": self.precision,
            "epochs": [e.to_dict() for e in self.epochs],
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "used_fallback_model": self.used_fallback_model,
            "dataset_summary": self.dataset_summary,
            "output_units": "relative (unitless) - NOT metres",
        }

    def summary(self) -> str:
        """A short human-readable report. Unmeasured things say so."""
        lines = [
            f"run_id            : {self.run_id}",
            f"device            : {self.device}",
            f"precision         : {self.precision.get('effective')}"
            + (
                f"  (requested {self.precision.get('requested')}, substituted: "
                f"{self.precision.get('substitution_reason')})"
                if self.precision.get("was_substituted")
                else ""
            ),
            f"encoder           : {self.model.get('encoder')} "
            f"[{self.model.get('weights_source')}]",
            f"trainable params  : {self.model.get('params_trainable')}",
            f"output units      : relative (unitless) - NOT metres",
        ]
        if self.used_fallback_model:
            lines.append(
                "WARNING           : encoder weights are RANDOM. Every number below "
                "describes random weights, not imagery."
            )
        for result in self.epochs:
            val = (
                f"{result.val_loss:.5f}" if result.val_loss is not None else "not measured"
            )
            lines.append(
                f"epoch {result.epoch:>3}       : train {result.train_loss:.5f}  "
                f"val {val}  steps {result.optimizer_steps}  {result.seconds:.1f}s"
            )
        evaluated = self.dataset_summary.get("val_samples")
        available = self.dataset_summary.get("val_samples_available")
        if evaluated and available:
            lines.append(
                f"val crops/epoch   : {evaluated} of {available}"
                + ("  (fixed subset; set training.max_val_crops: null for all)"
                   if evaluated < available else "  (full grid)")
            )
        if self.best_val_loss is None:
            lines.append("best val loss     : not measured (no validation split)")
        else:
            lines.append(
                f"best val loss     : {self.best_val_loss:.5f} (epoch {self.best_epoch})"
            )
        lines.append(
            "NOTE              : no real-world accuracy is claimed here. These are "
            "training-set and held-out losses on the configured dataset only."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Owns the model, the optimiser, the loaders and the checkpointing.

    Args:
        cfg: the full Phase 3 config.
        model: an already-built model. Defaults to
            :func:`~depthwizard.relative.model.build_model`.
        loaders: ``(train_loader, val_loader)``. Defaults to
            :func:`~depthwizard.relative.dataset.build_dataloaders`, which
            requires a real dataset on disk.
    """

    def __init__(
        self,
        cfg: RelativeConfig,
        *,
        model: RelativeHeightModel | None = None,
        loaders: tuple[DataLoader, DataLoader | None] | None = None,
    ) -> None:
        self.cfg = cfg
        set_seed(cfg.training.seed)

        self.device = resolve_device(cfg.training.device)
        self.precision = resolve_precision(cfg.training, self.device)

        self.model = (
            model
            if model is not None
            else build_model(cfg.model, output_size=cfg.training.image_size)
        )
        self.model.to(self.device)

        self.train_loader, self.val_loader = (
            loaders if loaders is not None else build_dataloaders(cfg)
        )

        trainable = self.model.trainable_parameters()
        if not trainable:
            raise RuntimeError(
                "the model reports no trainable parameters. With a frozen encoder "
                "the decoder and head must still be trainable; something has "
                "frozen everything."
            )
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, cfg.training.epochs)
            )
            if cfg.training.use_scheduler
            else None
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.precision.use_grad_scaler
        )

        self.start_epoch = 0
        #: Crops scored by the most recent validate() call.
        self.last_val_crops: int | None = None
        self.best_val_loss: float | None = None
        self.best_epoch: int | None = None
        if cfg.checkpoint.resume:
            self._resume(Path(cfg.checkpoint.resume))

        log.info(
            "trainer ready",
            extra={
                "device": str(self.device),
                "precision": self.precision.effective,
                "trainable_params": sum(p.numel() for p in trainable),
                "effective_batch_size": cfg.training.effective_batch_size,
            },
        )

    # -- autocast -----------------------------------------------------------

    def _autocast(self):
        if self.precision.autocast_dtype is None:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(
            device_type=self.device.type, dtype=self.precision.autocast_dtype
        )

    def _to_device(self, batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        return (
            batch["image"].to(self.device, non_blocking=True),
            batch["height"].to(self.device, non_blocking=True),
            batch["valid"].to(self.device, non_blocking=True),
        )

    # -- training -----------------------------------------------------------

    def train_epoch(self, epoch: int) -> EpochResult:
        """One pass over the training loader."""
        dataset = self.train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        started = time.perf_counter()
        accumulation = self.cfg.training.gradient_accumulation_steps
        max_steps = self.cfg.training.max_steps_per_epoch

        loss_total = 0.0
        scored_batches = 0
        micro_batches = 0
        skipped = 0
        optimizer_steps = 0
        pending = 0  # micro-batches whose gradients are not yet applied

        self.optimizer.zero_grad(set_to_none=True)

        for batch in self.train_loader:
            image, height, valid = self._to_device(batch)
            micro_batches += 1

            with self._autocast():
                prediction = self.model(image)
            output: LossOutput = scale_invariant_loss(
                prediction, height, valid, self.cfg.loss
            )

            if output.usable_count == 0:
                # Every sample fell below loss.min_valid_pixels. There is no
                # gradient to take here; taking one anyway would push the model
                # with a number computed from nothing.
                skipped += 1
                continue

            self.scaler.scale(output.loss / accumulation).backward()
            pending += 1
            loss_total += float(output.loss.detach().item())
            scored_batches += 1

            if pending >= accumulation:
                self._optimizer_step()
                optimizer_steps += 1
                pending = 0
                if max_steps is not None and optimizer_steps >= max_steps:
                    break

            if micro_batches % self.cfg.training.log_every == 0:
                log.info(
                    "train",
                    extra={
                        "epoch": epoch,
                        "micro_batch": micro_batches,
                        "steps": optimizer_steps,
                        "loss": round(float(output.loss.detach().item()), 5),
                        "usable": output.usable_count,
                        "dropped": output.stats.get("dropped_samples", 0),
                    },
                )

        if pending:
            # Leftover gradients from a partial accumulation group. Applying
            # them is what makes expected_steps() correct, and discarding them
            # would throw away real work at the end of every epoch.
            self._optimizer_step()
            optimizer_steps += 1

        if self.scheduler is not None:
            self.scheduler.step()

        peak_vram = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else None
        )
        return EpochResult(
            epoch=epoch,
            train_loss=(loss_total / scored_batches) if scored_batches else float("nan"),
            optimizer_steps=optimizer_steps,
            micro_batches=micro_batches,
            skipped_batches=skipped,
            seconds=time.perf_counter() - started,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            peak_vram_bytes=peak_vram,
        )

    def _optimizer_step(self) -> None:
        clip = self.cfg.training.grad_clip_norm
        if clip is not None:
            # Unscale first, or the clip threshold would be applied to gradients
            # that are still multiplied by the fp16 loss scale.
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.trainable_parameters(), clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    # -- validation ---------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple[float | None, dict[str, float] | None]:
        """Score the held-out scenes.

        Returns ``(None, None)`` when there is no validation loader, or when no
        batch in it had enough valid pixels to score. That is reported as "not
        measured" rather than as a number.

        The number of crops run through the model is logged and kept in
        :attr:`last_val_crops`; with ``training.max_val_crops`` set it is a
        fixed subset of the full validation grid, and the log says so.
        """
        self.last_val_crops = None
        if self.val_loader is None:
            return None, None

        self.model.eval()
        # The loss is a mean over usable samples, so it is pooled weighted by
        # usable_count: a batch with one usable tile must not count as much as
        # a batch with eight.
        loss_total = 0.0
        loss_weight = 0
        crops = 0
        aggregate = _MetricAggregate()

        for batch in self.val_loader:
            image, height, valid = self._to_device(batch)
            crops += int(image.shape[0])
            with self._autocast():
                prediction = self.model(image)
            output = scale_invariant_loss(prediction, height, valid, self.cfg.loss)
            if output.usable_count == 0:
                continue
            loss_total += float(output.loss.item()) * output.usable_count
            loss_weight += output.usable_count
            aggregate.add(relative_metrics(prediction, height, valid, self.cfg.loss))

        self.last_val_crops = crops
        available = full_length(self.val_loader.dataset)
        log.info(
            "validation",
            extra={
                "epoch": epoch,
                "val_crops_evaluated": crops,
                "val_crops_available": available,
                "val_crops_scored": loss_weight,
                "subset": crops < available,
            },
        )

        # The model is left in eval mode; train_epoch() switches it back.
        if loss_weight == 0:
            log.warning(
                "validation scored no batches: every validation tile fell below "
                "loss.min_valid_pixels. No validation number is reported for this "
                "epoch rather than a fabricated one.",
                extra={"epoch": epoch},
            )
            return None, None

        return loss_total / loss_weight, aggregate.result()

    # -- checkpoints --------------------------------------------------------

    def _state_to_save(self) -> dict[str, Any]:
        """The model state a checkpoint carries.

        By default the frozen encoder is excluded: it never changes, and the
        checkpoint records its name and weight source, so the full model is
        reconstructible from this state plus the config. ``include_encoder``
        keeps everything at the cost of ~90 MB per epoch.
        """
        state = self.model.state_dict()
        if self.cfg.checkpoint.include_encoder:
            return state
        return {key: value for key, value in state.items() if not key.startswith("encoder.")}

    def save_checkpoint(self, path: Path, result: EpochResult) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": result.epoch,
            "model_state": self._state_to_save(),
            "includes_encoder": self.cfg.checkpoint.include_encoder,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state": self.scaler.state_dict(),
            "config": self.cfg.to_dict(),
            "model_description": self.model.describe(),
            "precision": self.precision.to_dict(),
            "epoch_result": result.to_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "output_units": "relative (unitless) - NOT metres",
        }
        torch.save(payload, path)
        log.info("wrote checkpoint", extra={"path": str(path), "epoch": result.epoch})
        return path

    def _resume(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"checkpoint.resume points at {path}, which does not exist. "
                "Set checkpoint.resume: null to start from scratch."
            )
        payload = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(
            payload["model_state"], strict=False
        )
        if not payload.get("includes_encoder", False):
            # The encoder is expected to be missing; anything else is not.
            missing = [key for key in missing if not key.startswith("encoder.")]
        if missing or unexpected:
            raise RuntimeError(
                f"{path}: checkpoint does not match this model. "
                f"missing={missing[:5]} unexpected={unexpected[:5]}. "
                "Refusing to resume from a partially-matching checkpoint."
            )
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler is not None and payload.get("scheduler_state"):
            self.scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state"):
            self.scaler.load_state_dict(payload["scaler_state"])
        self.start_epoch = int(payload["epoch"]) + 1
        self.best_val_loss = payload.get("best_val_loss")
        self.best_epoch = payload.get("best_epoch")
        log.info(
            "resumed from checkpoint",
            extra={"path": str(path), "next_epoch": self.start_epoch},
        )

    # -- the run ------------------------------------------------------------

    def fit(self) -> TrainingReport:
        """Train for ``training.epochs`` epochs and return the run's record."""
        from depthwizard.relative.fallback import is_fallback

        report = TrainingReport(
            run_id=self.cfg.run_id,
            config=self.cfg.to_dict(),
            model=self.model.describe(),
            device=str(self.device),
            precision=self.precision.to_dict(),
            used_fallback_model=is_fallback(self.model),
            dataset_summary={
                "train_samples": len(self.train_loader.dataset),
                # Crops scored per epoch, and the full grid they are drawn from.
                "val_samples": (
                    len(self.val_loader.dataset) if self.val_loader is not None else 0
                ),
                "val_samples_available": (
                    full_length(self.val_loader.dataset)
                    if self.val_loader is not None
                    else 0
                ),
                "max_val_crops": self.cfg.training.max_val_crops,
                "split_mode": self.cfg.dataset.split.mode,
                "spatially_separated": self.cfg.dataset.split.is_spatially_separated,
            },
        )

        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            result = self.train_epoch(epoch)
            result.val_loss, result.val_metrics = self.validate(epoch)
            result.val_crops = self.last_val_crops
            report.epochs.append(result)

            if result.val_loss is not None and (
                self.best_val_loss is None or result.val_loss < self.best_val_loss
            ):
                self.best_val_loss = result.val_loss
                self.best_epoch = epoch
                if self.cfg.checkpoint.keep_best:
                    self.save_checkpoint(self.cfg.checkpoint.dir / "best.pt", result)

            if self.cfg.checkpoint.every_epoch:
                self.save_checkpoint(
                    self.cfg.checkpoint.dir / f"epoch_{epoch:03d}.pt", result
                )

            log.info("epoch complete", extra=_loggable(result.to_dict()))

        report.best_val_loss = self.best_val_loss
        report.best_epoch = self.best_epoch
        return report


#: Totals across the split; summed, never averaged.
_COUNT_METRICS = frozenset({"valid_pixels", "scored_samples"})
#: Constant markers about how the metrics were computed; carried through as-is.
_FLAG_METRICS = frozenset({"scale_aligned"})
#: Per-sample means inside a batch; pooled weighted by ``scored_samples``.
_SAMPLE_METRICS = frozenset({"si_rmse", "rmse_log_aligned"})
# Everything else (abs_rel, delta_*) is pooled over pixels inside a batch and
# is weighted by ``valid_pixels``.


class _MetricAggregate:
    """Pools per-batch :func:`relative_metrics` into split-level numbers.

    Each metric is weighted by what it was actually averaged over inside the
    batch, so the pooled value equals the one a single pass over the whole
    split would give. A batch whose metrics are ``nan`` (nothing valid)
    contributes nothing, and a split with nothing scored returns ``None``.
    """

    def __init__(self) -> None:
        self._weighted: dict[str, float] = {}
        self._weights: dict[str, float] = {}
        self._counts: dict[str, float] = {}
        self._flags: dict[str, float] = {}

    def add(self, metrics: dict[str, float]) -> None:
        samples = metrics.get("scored_samples", 0.0)
        pixels = metrics.get("valid_pixels", 0.0)
        for key, value in metrics.items():
            if key in _COUNT_METRICS:
                self._counts[key] = self._counts.get(key, 0.0) + value
            elif key in _FLAG_METRICS:
                self._flags[key] = value
            elif not math.isnan(value):
                weight = samples if key in _SAMPLE_METRICS else pixels
                if weight > 0:
                    self._weighted[key] = self._weighted.get(key, 0.0) + value * weight
                    self._weights[key] = self._weights.get(key, 0.0) + weight

    def result(self) -> dict[str, float] | None:
        if not self._weights:
            return None
        pooled = {key: total / self._weights[key] for key, total in self._weighted.items()}
        return {**pooled, **self._counts, **self._flags}


def _loggable(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten a result dict to scalars the structured logger can render."""
    return {
        key: value
        for key, value in data.items()
        if value is None or isinstance(value, (int, float, str, bool))
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    """``python -m depthwizard.relative.train --config configs/phase3.yaml``."""
    import argparse

    from depthwizard.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Train the Phase 3 relative-height baseline")
    parser.add_argument("--config", default="configs/phase3.yaml")
    parser.add_argument("--log-format", choices=("text", "json"), default="text")
    args = parser.parse_args(list(argv) if argv is not None else None)

    setup_logging({"level": "INFO", "format": args.log_format})
    cfg = load_relative_config(args.config)

    trainer = Trainer(cfg)
    report = trainer.fit()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.output_dir / f"{cfg.run_id}_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print(report.summary())
    print(f"\nreport written to {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
