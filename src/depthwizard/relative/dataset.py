"""The thin PyTorch layer over :mod:`depthwizard.relative.data`.

Everything that *decides* what a training sample is -- pair discovery, the
spatial split, crop geometry, paired reading, the validity mask -- lives in
:mod:`depthwizard.relative.data` and is pure numpy. This module only turns those
decisions into tensors and hands them to a ``DataLoader``. Keeping the split
means the interesting logic stays testable without importing torch, and this
file stays small enough to audit at a glance.

Determinism
-----------
Training crops are random, but never *unreproducibly* random. The RNG for
sample ``i`` of epoch ``e`` is seeded from ``(dataset.seed, e, i)``, so:

* two runs with the same config see the same crops in the same order;
* a given sample's crop does not depend on how many workers are running, on
  which worker picked it up, or on the order the loader happened to schedule
  it -- the usual way a multi-worker pipeline stops being reproducible;
* each epoch still sees different crops, because ``e`` is in the seed.

:meth:`RelativeHeightDataset.set_epoch` is what advances ``e``, and the trainer
calls it before every epoch. Forgetting to call it does not crash, it just
re-shows epoch 0's crops -- so the trainer logs the epoch seed it is using.

The epoch lives in a one-element **shared-memory** tensor, not a plain int.
With ``persistent_workers=True`` each DataLoader worker keeps the copy of the
dataset it was handed when it started, so a plain attribute set in the main
process after that is never seen by the workers -- every epoch would silently
replay epoch 0's crops. A shared-memory tensor is passed to the workers as a
handle to the same storage, so ``set_epoch`` in the main process is visible to
every worker on its next ``__getitem__``. The trainer calls ``set_epoch``
before creating the epoch's iterator, and workers only fetch after that.

Validation crops are not random at all: they are the full non-overlapping tile
grid of each validation scene, in raster order, identical on every epoch and
every machine. Two validation losses are therefore comparable.

No data ships here
------------------
This module reads only what ``dataset.root`` points at. DFC2019 is not included
in this repository, is not downloaded by it, and is not simulated by it. With no
dataset present, constructing a dataset raises
:class:`~depthwizard.relative.data.DatasetError` naming the missing path.
"""

from __future__ import annotations

import random
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from depthwizard.logging_setup import get_logger
from depthwizard.relative.config import DatasetConfig, RelativeConfig, TrainingConfig
from depthwizard.relative.data import (
    CropWindow,
    DatasetError,
    ScenePair,
    TileSample,
    check_pair_registration,
    discover_pairs,
    plan_train_crops,
    plan_validation_crops,
    raster_shape,
    read_pair_tile,
    split_pairs,
)

__all__ = [
    "RelativeHeightDataset",
    "collate_samples",
    "build_datasets",
    "build_dataloaders",
    "limit_validation",
    "full_length",
    "sample_to_tensors",
]

log = get_logger(__name__)


def sample_to_tensors(sample: TileSample) -> dict[str, Any]:
    """One :class:`TileSample` -> the tensor dict a batch is made of.

    ``image`` is ``[3, H, W]`` float32, ``height`` and ``valid`` are ``[1, H, W]``
    -- float32 and bool respectively. The channel axis on the target matches the
    model's ``[B, 1, H, W]`` output so the loss never has to guess which of the
    two conventions it was handed.
    """
    return {
        "image": torch.from_numpy(np.ascontiguousarray(sample.image)),
        "height": torch.from_numpy(np.ascontiguousarray(sample.height))[None],
        "valid": torch.from_numpy(np.ascontiguousarray(sample.valid))[None],
        "metadata": sample.metadata,
    }


def collate_samples(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of samples, keeping metadata as a plain list of dicts.

    The default collate would try to tensorise the metadata, turning a CRS
    string into something unhelpful and a per-sample transform into a
    transposed batch of floats. Metadata is bookkeeping, not model input, so it
    is passed through untouched and stays readable at the other end.
    """
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "height": torch.stack([item["height"] for item in batch]),
        "valid": torch.stack([item["valid"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }


class RelativeHeightDataset(Dataset):
    """Paired (RGB, height, valid) crops from a set of DFC2019-style scenes.

    Args:
        pairs: the scenes this dataset draws from -- already split, so a
            dataset instance is either train or validation and can never mix
            the two.
        cfg: dataset configuration.
        train: True for random crops (``train_tiles_per_scene`` per scene per
            epoch), False for the deterministic full tile grid.
        epoch: initial epoch index, folded into every training crop's seed.
    """

    def __init__(
        self,
        pairs: Sequence[ScenePair],
        cfg: DatasetConfig,
        *,
        train: bool,
        epoch: int = 0,
    ) -> None:
        if not pairs:
            raise DatasetError(
                "cannot build a dataset from zero scenes. Check dataset.root and "
                "the split configuration against the scenes actually on disk."
            )
        self.pairs = tuple(pairs)
        self.cfg = cfg
        self.train = bool(train)
        # Shared with DataLoader workers, including persistent ones; see the
        # module docstring. Written only by set_epoch().
        self._shared_epoch = torch.tensor([int(epoch)], dtype=torch.int64).share_memory_()

        # Raster shapes are read once from the headers here rather than per
        # __getitem__, so crop planning costs nothing at sample time. This also
        # surfaces an unreadable raster at construction rather than midway
        # through the first epoch.
        self._shapes: dict[str, tuple[int, int]] = {}
        not_georeferenced = 0
        for pair in self.pairs:
            try:
                self._shapes[pair.stem] = raster_shape(pair.image_path)
            except Exception as exc:  # rasterio raises a family of errors here
                raise DatasetError(
                    f"could not read the header of {pair.image_path}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            # CRS / transform / footprint agreement (or, with no CRS, an exact
            # grid match), checked up front so a misregistered scene fails
            # here, not midway through an epoch.
            if not check_pair_registration(pair):
                not_georeferenced += 1

        # Reported once here rather than on every tile read: for the real
        # DFC2019 release this is every pair, and it is expected.
        if not_georeferenced:
            log.info(
                "pairs without a CRS: registration was checked as an exact pixel-grid "
                "match (identical size and transform); no footprint check is possible "
                "and no resampling is allowed for them",
                extra={
                    "split": "train" if train else "val",
                    "not_georeferenced_pairs": not_georeferenced,
                    "pairs": len(self.pairs),
                },
            )

        if self.train:
            # index -> pair; each scene contributes train_tiles_per_scene entries.
            self._index: tuple[tuple[int, CropWindow | None], ...] = tuple(
                (pair_index, None)
                for pair_index in range(len(self.pairs))
                for _ in range(cfg.train_tiles_per_scene)
            )
        else:
            plan: list[tuple[int, CropWindow | None]] = []
            for pair_index, pair in enumerate(self.pairs):
                for window in plan_validation_crops(self._shapes[pair.stem], cfg):
                    plan.append((pair_index, window))
            self._index = tuple(plan)

        log.info(
            "built relative-height dataset",
            extra={
                "split": "train" if self.train else "val",
                "scenes": len(self.pairs),
                "samples": len(self._index),
                "tile_size": cfg.tile_size,
            },
        )

    # -- epoch ---------------------------------------------------------------

    @property
    def epoch(self) -> int:
        """The current epoch, as seen by this process (main or worker)."""
        return int(self._shared_epoch[0].item())

    def set_epoch(self, epoch: int) -> None:
        """Advance the training-crop seed, in every worker. A no-op for validation."""
        self._shared_epoch[0] = int(epoch)

    def _rng(self, index: int) -> random.Random:
        """Deterministic per-(epoch, index) RNG, independent of worker layout."""
        # The multipliers are arbitrary large odd numbers; they only need to
        # keep (epoch, index) pairs from colliding on the same seed.
        return random.Random(
            (self.cfg.seed * 1_000_003) ^ (self.epoch * 9_176_111) ^ index
        )

    # -- Dataset protocol ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair_index, window = self._index[index]
        pair = self.pairs[pair_index]
        shape = self._shapes[pair.stem]

        if window is None:
            rng = self._rng(index)
            sample = self._sample_train_tile(pair, shape, rng)
        else:
            sample = read_pair_tile(pair, window, self.cfg)

        tensors = sample_to_tensors(sample)
        tensors["metadata"]["split"] = "train" if self.train else "val"
        tensors["metadata"]["epoch"] = self.epoch
        tensors["metadata"]["index"] = int(index)
        return tensors

    def _sample_train_tile(
        self, pair: ScenePair, shape: tuple[int, int], rng: random.Random
    ) -> TileSample:
        """Draw a crop, retrying mostly-nodata windows.

        Mirrors :func:`depthwizard.relative.data.sample_train_tile`, but takes
        the RNG this dataset constructed so the draw stays tied to
        ``(seed, epoch, index)`` rather than to global RNG state that a
        DataLoader worker would fork unpredictably.
        """
        best: TileSample | None = None
        for _ in range(self.cfg.max_crop_attempts):
            window = plan_train_crops(pair, shape, self.cfg, rng)
            sample = read_pair_tile(pair, window, self.cfg)
            if sample.valid_fraction >= self.cfg.min_valid_fraction:
                return sample
            if best is None or sample.valid_fraction > best.valid_fraction:
                best = sample
        assert best is not None  # max_crop_attempts >= 1, enforced by the config
        return best

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]


# ---------------------------------------------------------------------------
# Construction from a config
# ---------------------------------------------------------------------------


def build_datasets(
    cfg: RelativeConfig,
) -> tuple[RelativeHeightDataset, RelativeHeightDataset | None]:
    """Discover, split and wrap the configured dataset.

    Returns ``(train, val)``; ``val`` is None when the split held nothing out,
    which :func:`depthwizard.relative.data.split_pairs` has already warned
    about. A warning is emitted here too when the split is not spatially
    separated, because a validation score from such a split does not mean what
    a reader will assume it means.
    """
    pairs = discover_pairs(cfg.dataset)
    train_pairs, val_pairs = split_pairs(pairs, cfg.dataset.split)

    if not cfg.dataset.split.is_spatially_separated:
        log.warning(
            "the configured split is NOT spatially separated; any validation "
            "number it produces measures interpolation between neighbouring "
            "tiles, not generalisation to unseen ground, and must be reported "
            "with that caveat",
            extra={"mode": cfg.dataset.split.mode},
        )

    train = RelativeHeightDataset(train_pairs, cfg.dataset, train=True)
    val = (
        RelativeHeightDataset(val_pairs, cfg.dataset, train=False) if val_pairs else None
    )
    return train, val


def limit_validation(
    val_ds: RelativeHeightDataset, max_crops: int | None, seed: int
) -> Dataset:
    """A fixed, reproducible subset of at most ``max_crops`` validation crops.

    ``max_crops`` of ``None``, or at least ``len(val_ds)``, returns ``val_ds``
    itself: the full grid. Otherwise the indices are sampled once from
    ``random.Random(seed)`` without replacement and kept in raster order, so
    the same crops are scored on every epoch, every run and every machine, and
    they are spread over all validation scenes and grid positions (not, as a
    plain prefix would be, the first few scenes only). Each sample keeps its
    index into the full grid in ``metadata["index"]``. The full dataset stays
    reachable as ``subset.dataset``.
    """
    total = len(val_ds)
    if max_crops is None or max_crops >= total:
        return val_ds
    indices = sorted(random.Random(seed).sample(range(total), max_crops))
    return Subset(val_ds, indices)


def full_length(dataset: Dataset) -> int:
    """Length of the dataset beneath any :class:`~torch.utils.data.Subset`."""
    return len(dataset.dataset) if isinstance(dataset, Subset) else len(dataset)


def build_dataloaders(
    cfg: RelativeConfig,
    *,
    datasets: tuple[RelativeHeightDataset, RelativeHeightDataset | None] | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    """Wrap the datasets in loaders sized by ``cfg.training``.

    ``datasets`` lets a caller (notably the tests) supply datasets built from
    something other than :func:`build_datasets`. Nothing else differs.

    The validation loader scores at most ``training.max_val_crops`` crops per
    epoch (see :func:`limit_validation`); ``null`` scores the full grid.
    """
    train_ds, val_ds = datasets if datasets is not None else build_datasets(cfg)
    training: TrainingConfig = cfg.training
    if val_ds is not None:
        full_val_ds = val_ds
        val_ds = limit_validation(val_ds, training.max_val_crops, cfg.dataset.seed)
        log.info(
            "validation crops per epoch",
            extra={
                "evaluated": len(val_ds),
                "available": len(full_val_ds),
                "limited": val_ds is not full_val_ds,
                "max_val_crops": training.max_val_crops,
                "seed": cfg.dataset.seed,
            },
        )

    loader_kwargs: dict[str, Any] = {
        "num_workers": training.num_workers,
        "pin_memory": training.pin_memory,
        "collate_fn": collate_samples,
    }
    if training.num_workers > 0:
        loader_kwargs["persistent_workers"] = training.persistent_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=training.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=training.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        if val_ds is not None
        else None
    )
    return train_loader, val_loader
