"""Opt-in smoke test against a real local DFC2019 Track-1 download.

Skipped unless BOTH hold:

* ``DEPTHWIZARD_REAL_DATA=1`` is set (reading ~5,600 raster headers takes a
  while, so it never runs by accident), and
* ``configs/phase3.local.yaml`` exists and its ``dataset.root`` is present.

Run with::

    DEPTHWIZARD_REAL_DATA=1 pytest -m real_data

It checks the dataset's shape and that the loader accepts it -- pair and tile
counts, the no-CRS exact-grid registration of every pair, the per-city split,
one real tile read. It trains nothing and reports no accuracy. The expected
counts are those of the verified training release (2,783 pairs, 108 tile ids);
a different download failing them is the point.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from depthwizard.relative.config import load_relative_config
from depthwizard.relative.data import (
    CropWindow,
    check_pair_registration,
    discover_pairs,
    raster_shape,
    read_pair_tile,
    split_pairs,
)

pytestmark = pytest.mark.real_data

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = REPO_ROOT / "configs" / "phase3.local.yaml"

EXPECTED_PAIRS = 2783
EXPECTED_TILE_IDS = 108


def _local_config():
    if os.environ.get("DEPTHWIZARD_REAL_DATA") != "1":
        pytest.skip("real-data smoke test is opt-in: set DEPTHWIZARD_REAL_DATA=1")
    if not LOCAL_CONFIG.is_file():
        pytest.skip("configs/phase3.local.yaml not found")
    cfg = load_relative_config(LOCAL_CONFIG)
    if not cfg.dataset.root.is_dir():
        pytest.skip(f"DFC2019 not present at {cfg.dataset.root}")
    return cfg


@pytest.fixture(scope="module")
def real_cfg():
    return _local_config()


@pytest.fixture(scope="module")
def real_pairs(real_cfg):
    return discover_pairs(real_cfg.dataset)


def test_real_dataset_has_every_pair_and_tile_id(real_pairs):
    assert len(real_pairs) == EXPECTED_PAIRS
    assert len({p.scene_id for p in real_pairs}) == EXPECTED_TILE_IDS
    assert {p.city for p in real_pairs} == {"JAX", "OMA"}


def test_every_real_pair_passes_exact_grid_registration(real_pairs):
    """All real pairs are CRS-less; each must match size and transform exactly."""
    georeferenced = [check_pair_registration(p) for p in real_pairs]
    assert not any(georeferenced)


def test_real_per_city_split(real_cfg, real_pairs):
    train, val = split_pairs(real_pairs, real_cfg.dataset.split)
    assert len(train) + len(val) == EXPECTED_PAIRS
    assert not {p.scene_id for p in train} & {p.scene_id for p in val}
    assert {p.city for p in train} == {p.city for p in val} == {"JAX", "OMA"}
    for city in ("JAX", "OMA"):
        tiles = {p.scene_id for p in real_pairs if p.city == city}
        val_tiles = {p.scene_id for p in val if p.city == city}
        assert len(val_tiles) == round(real_cfg.dataset.split.val_fraction * len(tiles))
    train_stems = {p.stem for p in train}
    assert ("JAX_004_006" in train_stems) == ("JAX_004_007" in train_stems)


def test_real_validation_is_bounded_to_a_fixed_subset(real_cfg, real_pairs):
    pytest.importorskip("torch")
    from depthwizard.relative.dataset import (
        RelativeHeightDataset,
        full_length,
        limit_validation,
    )

    _, val = split_pairs(real_pairs, real_cfg.dataset.split)
    full = RelativeHeightDataset(val, real_cfg.dataset, train=False)
    assert len(full) == len(val) * 16  # 4x4 grid of 256 px crops per 1024 px pair
    limit = real_cfg.training.max_val_crops
    subset = limit_validation(full, limit, real_cfg.dataset.seed)
    assert len(subset) == min(limit, len(full))
    assert full_length(subset) == len(full)
    again = limit_validation(full, limit, real_cfg.dataset.seed)
    assert list(again.indices) == list(subset.indices)


def test_one_real_pair_reads_with_a_valid_mask(real_cfg, real_pairs):
    pair = next(p for p in real_pairs if p.stem == "JAX_004_006")
    assert raster_shape(pair.image_path) == (1024, 1024)
    size = real_cfg.dataset.tile_size
    sample = read_pair_tile(pair, CropWindow(384, 384, size), real_cfg.dataset)
    assert sample.image.shape == (3, size, size)
    assert sample.valid.any()
    assert np.isfinite(sample.height).all()
    assert sample.metadata["georeferenced"] is False
    assert sample.metadata["crs"] == ""
    assert sample.metadata["height_resampled"] is False
