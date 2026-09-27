"""Phase 5 geographic split audit, config strictness, and the confidence proxy.

The split tests use DFC2019-SHAPED synthetic rasters (``conftest.write_synthetic_pair``):
only filenames and layout matter to a split, never pixel values.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from depthwizard.config import ConfigError
from depthwizard.relative.config import DatasetConfig, SplitConfig
from depthwizard.relative.data import discover_pairs
from depthwizard.validation.confidence import NOT_AVAILABLE, ConfidenceState, confidence_proxy
from depthwizard.validation.config import (
    Phase5Config,
    RegionSplitConfig,
    load_phase5_config,
)
from depthwizard.validation.spatial_split import SplitError, audit_region_split, region_of, select_regions

REPO_ROOT = Path(__file__).resolve().parents[1]
JAX_TRAIN_OMA_HELD = RegionSplitConfig(train_regions=("JAX",), heldout_regions=("OMA",))
OMA_HELD_OUT = SplitConfig(mode="scene_prefix", val_scene_prefixes=("OMA",))


def _discover(root):
    return discover_pairs(DatasetConfig(root=root, split=OMA_HELD_OUT))


@pytest.fixture
def pairs(dfc_like_root):
    return _discover(dfc_like_root)


# ---------------------------------------------------------------------------
# Spatial split
# ---------------------------------------------------------------------------


def test_correct_region_separation(pairs):
    audit = audit_region_split(JAX_TRAIN_OMA_HELD, pairs, OMA_HELD_OUT)
    assert audit.train_regions == ("JAX",) and audit.heldout_regions == ("OMA",)
    assert audit.checkpoint_train_pairs_by_region == {"JAX": 2}
    assert audit.heldout_pairs_by_region == {"OMA": 2}
    assert audit.heldout_scene_ids == ("OMA_012",)
    record = audit.to_dict()
    assert record["random_split"] is False and record["scene_overlap"] == 0


def test_heldout_region_isolation_selects_only_that_region(pairs):
    held = select_regions(pairs, ["OMA"])
    assert {p.city for p in held} == {"OMA"} and len(held) == 2
    assert not {p.scene_id for p in held} & {p.scene_id for p in select_regions(pairs, ["JAX"])}


def test_leakage_of_heldout_city_into_training_is_refused(pairs):
    # per_city_scene trains on every city (here each city has one scene, so both train).
    with pytest.raises(SplitError, match=r"held-out region\(s\) \['OMA'\] were used to train"):
        audit_region_split(JAX_TRAIN_OMA_HELD, pairs, SplitConfig(mode="per_city_scene"))


def test_swapped_declaration_is_refused(pairs):
    declared = RegionSplitConfig(train_regions=("OMA",), heldout_regions=("JAX",))
    with pytest.raises(SplitError, match="JAX"):
        audit_region_split(declared, pairs, OMA_HELD_OUT)


@pytest.mark.parametrize("mode", ["random_tile", "random_scene"])
def test_random_split_cannot_replace_the_spatial_split(pairs, mode):
    with pytest.raises(SplitError, match="NOT spatially separated"):
        audit_region_split(JAX_TRAIN_OMA_HELD, pairs, SplitConfig(mode=mode, val_fraction=0.5))


def test_undeclared_training_region_is_refused(tmp_path, synthetic_pair_writer):
    root = tmp_path / "three_cities"
    for stem in ("JAX_004_001", "OMA_012_001", "ATL_001_001"):
        synthetic_pair_writer(root, stem, size=32)
    pairs = _discover(root)
    with pytest.raises(SplitError, match=r"\['ATL'\].*does not declare"):
        audit_region_split(JAX_TRAIN_OMA_HELD, pairs, OMA_HELD_OUT)


def test_declared_training_region_that_never_trained_is_refused(pairs):
    declared = RegionSplitConfig(train_regions=("JAX", "ATL"), heldout_regions=("OMA",))
    with pytest.raises(SplitError, match="contributed no training pair"):
        audit_region_split(declared, pairs, OMA_HELD_OUT)


# -- model-selection leakage ---------------------------------------------------
# Real Phase 3 metadata shape: best.pt = epoch 6 / best_epoch 6; epoch_009.pt =
# epoch 9 / best_epoch 6; training.epochs = 10.


def _selection(epoch, best_epoch=6, epochs=10):
    from depthwizard.validation.spatial_split import CheckpointSelection

    return CheckpointSelection.from_payload(
        {"epoch": epoch, "best_epoch": best_epoch, "config": {"training": {"epochs": epochs}}})


def test_heldout_city_on_the_validation_side_is_recorded(pairs):
    audit = audit_region_split(JAX_TRAIN_OMA_HELD, pairs, OMA_HELD_OUT)
    assert audit.checkpoint_validation_regions == ("OMA",)
    assert audit.to_dict()["checkpoint_validation_regions"] == ["OMA"]


def test_best_epoch_selected_on_the_heldout_city_is_refused(pairs):
    from depthwizard.validation.spatial_split import audit_model_selection

    audit = audit_region_split(JAX_TRAIN_OMA_HELD, pairs, OMA_HELD_OUT)
    with pytest.raises(SplitError, match=r"model-selection leakage.*best-validation-loss epoch.*\['OMA'\].*epoch_009\.pt"):
        audit_model_selection(_selection(epoch=6, best_epoch=6), audit)


def test_any_non_final_epoch_is_refused_when_validation_is_heldout(pairs):
    from depthwizard.validation.spatial_split import audit_model_selection

    audit = audit_region_split(JAX_TRAIN_OMA_HELD, pairs, OMA_HELD_OUT)
    with pytest.raises(SplitError, match="not the final pre-set epoch"):
        audit_model_selection(_selection(epoch=3, best_epoch=6), audit)


def test_final_pre_set_epoch_is_accepted(pairs):
    from depthwizard.validation.spatial_split import audit_model_selection

    audit = audit_region_split(JAX_TRAIN_OMA_HELD, pairs, OMA_HELD_OUT)
    record = audit_model_selection(_selection(epoch=9, best_epoch=6), audit)
    assert record["status"].startswith("PASSED") and record["final_epoch"] == 9
    # best.pt whose best epoch IS the final epoch holds the same weights.
    assert audit_model_selection(_selection(epoch=9, best_epoch=9), audit)["status"].startswith("PASSED")


def test_unverifiable_selection_is_refused(pairs):
    from depthwizard.validation.spatial_split import CheckpointSelection, audit_model_selection

    audit = audit_region_split(JAX_TRAIN_OMA_HELD, pairs, OMA_HELD_OUT)
    with pytest.raises(SplitError, match="cannot be verified"):
        audit_model_selection(CheckpointSelection.from_payload({"config": {}}), audit)


def test_selection_check_does_not_apply_when_validation_side_is_not_heldout(pairs):
    from dataclasses import replace

    from depthwizard.validation.spatial_split import audit_model_selection

    audit = replace(audit_region_split(JAX_TRAIN_OMA_HELD, pairs, OMA_HELD_OUT), checkpoint_validation_regions=("JAX",))
    record = audit_model_selection(_selection(epoch=6, best_epoch=6), audit)
    assert record["status"].startswith("not applicable")


def test_training_leakage_is_still_refused_before_selection(pairs):
    # The shipped situation: trained on OMA AND best-epoch selected. The training
    # leakage refusal comes first and is unchanged.
    with pytest.raises(SplitError, match="were used to train"):
        audit_region_split(JAX_TRAIN_OMA_HELD, pairs, SplitConfig(mode="per_city_scene"))


def test_pair_without_region_identifier_is_refused():
    with pytest.raises(SplitError, match="no city identifier"):
        region_of(SimpleNamespace(stem="mystery", city=""))


def test_split_config_requires_disjoint_named_regions():
    with pytest.raises(ConfigError, match="both training and held out"):
        RegionSplitConfig(train_regions=("JAX",), heldout_regions=("JAX",))
    with pytest.raises(ConfigError, match="non-empty"):
        RegionSplitConfig(train_regions=("JAX",), heldout_regions=())
    with pytest.raises(ConfigError, match="region_key"):
        RegionSplitConfig(region_key="tile", train_regions=("A",), heldout_regions=("B",))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_shipped_phase5_config_loads_and_declares_the_split():
    cfg = load_phase5_config(REPO_ROOT / "configs" / "phase5.yaml")
    assert cfg.dfc.split.train_regions == ("JAX",) and cfg.dfc.split.heldout_regions == ("OMA",)
    assert cfg.sun_band.min_deg == 25.0 and cfg.sun_band.max_deg == 45.0
    assert cfg.delta_thresholds == pytest.approx((1.25, 1.25**2, 1.25**3))


def test_config_is_strict_and_has_no_random_split_option():
    with pytest.raises(ConfigError):
        Phase5Config.from_dict({"dfc": {"split": {"mode": "random", "train_regions": ["A"], "heldout_regions": ["B"]}}})
    with pytest.raises(ConfigError):
        Phase5Config.from_dict({"not_a_key": 1})
    with pytest.raises(ConfigError, match="split"):
        Phase5Config.from_dict({"dfc": {"phase4a_dir": "x"}})
    assert Phase5Config().dfc is None  # no split is ever invented by default


# ---------------------------------------------------------------------------
# Confidence proxy
# ---------------------------------------------------------------------------

SHAPE = (4, 4)


_CLEAR = "clear"  # an available mask with no pixel flagged


def _proxy(sun=35.0, *, shadow=_CLEAR, water=_CLEAR, valid=None):
    """``shadow`` / ``water``: _CLEAR, a mask, or None (= input unavailable)."""
    def mask(value):
        return np.zeros(SHAPE, dtype=bool) if isinstance(value, str) else value

    valid = np.ones(SHAPE, dtype=bool) if valid is None else valid
    return confidence_proxy(valid, shadow_mask=mask(shadow), water_mask=mask(water), sun_elevation_deg=sun)


@pytest.mark.parametrize("elevation", [25.0, 35.0, 45.0])
def test_sun_inside_band_is_high(elevation):
    result = _proxy(elevation)
    assert (result.state == ConfidenceState.HIGH).all()
    assert result.diagnostics["checks"]["sun_band"]["sun_band"] == "within_band"


@pytest.mark.parametrize("elevation, band", [(15.0, "below_band"), (24.9, "below_band"),
                                             (45.1, "above_band"), (60.0, "above_band")])
def test_sun_outside_band_reduces_every_valid_pixel(elevation, band):
    result = _proxy(elevation)
    assert (result.state == ConfidenceState.REDUCED).all()
    check = result.diagnostics["checks"]["sun_band"]
    assert check["sun_band"] == band and check["flagged_valid_pixels"] == 16
    assert "outside recommended 25-45 degree band" in check["reason"]


def test_shadow_occlusion_reduces_confidence():
    shadow = np.zeros(SHAPE, dtype=bool)
    shadow[0, :2] = True
    result = _proxy(shadow=shadow)
    assert (result.state[0, :2] == ConfidenceState.REDUCED).all()
    assert (result.state[1:] == ConfidenceState.HIGH).all()
    assert result.diagnostics["checks"]["shadow_occlusion"]["flagged_valid_pixels"] == 2


def test_water_is_unsuitable_and_overrides_shadow():
    water = np.zeros(SHAPE, dtype=bool)
    water[2, :] = True
    shadow = np.zeros(SHAPE, dtype=bool)
    shadow[2, 0] = shadow[3, 0] = True
    result = _proxy(shadow=shadow, water=water)
    assert (result.state[2] == ConfidenceState.UNSUITABLE).all()
    assert result.state[3, 0] == ConfidenceState.REDUCED
    assert result.diagnostics["checks"]["water"]["flagged_valid_pixels"] == 4


def test_multiple_conditions_follow_the_documented_precedence():
    valid = np.ones(SHAPE, dtype=bool)
    valid[0, 0] = False
    water = np.zeros(SHAPE, dtype=bool)
    water[0, 0] = water[0, 1] = True
    shadow = np.ones(SHAPE, dtype=bool)
    result = _proxy(sun=70.0, shadow=shadow, water=water, valid=valid)
    assert result.state[0, 0] == ConfidenceState.INVALID     # invalid beats water
    assert result.state[0, 1] == ConfidenceState.UNSUITABLE  # water beats shadow + sun
    assert (result.state[1:] == ConfidenceState.REDUCED).all()
    counts = result.diagnostics["state_pixels"]
    assert counts == {"INVALID": 1, "UNSUITABLE": 1, "REDUCED": 14, "NOT_ASSESSED": 0, "HIGH": 0}
    assert result.diagnostics["overlaps"]["shadow_and_sun_out_of_band_valid_pixels"] == 14


def test_unavailable_inputs_are_never_counted_as_zero():
    result = _proxy(sun=None, water=None)
    assert (result.state == ConfidenceState.NOT_ASSESSED).all()  # cannot be HIGH
    checks = result.diagnostics["checks"]
    for name in ("water", "sun_band"):
        assert checks[name]["available"] is False
        assert checks[name]["flagged_valid_pixels"] == NOT_AVAILABLE
    assert result.diagnostics["all_checks_available"] is False


def test_confidence_rejects_misaligned_masks():
    with pytest.raises(ValueError, match="shadow"):
        confidence_proxy(np.ones(SHAPE, dtype=bool), shadow_mask=np.ones((2, 2), dtype=bool),
                         water_mask=None, sun_elevation_deg=None)
