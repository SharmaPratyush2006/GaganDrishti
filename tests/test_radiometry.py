"""Tests for percentile-based radiometric normalisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from depthwizard.ingest.loaders import GeoTIFFLoader
from depthwizard.ingest.radiometry import (
    apply_stretch,
    compute_band_stretches,
    normalize_to_uint8,
)


def test_uint16_input_uses_the_full_8bit_range() -> None:
    """12-bit-ish imagery would render near-black without a stretch."""
    rng = np.random.default_rng(0)
    array = (rng.random((64, 64, 1)) * 4095).astype(np.uint16)

    result = normalize_to_uint8(array)

    assert result.array.dtype == np.uint8
    assert result.array.min() == 0
    assert result.array.max() == 255


def test_percentile_window_is_recorded_not_just_applied() -> None:
    array = np.linspace(100, 1100, 101, dtype=np.float32).reshape(101, 1, 1)
    result = normalize_to_uint8(array, lower_percentile=2.0, upper_percentile=98.0)

    (stretch,) = result.bands
    assert stretch.band == 0
    assert stretch.lower_percentile == 2.0
    assert stretch.upper_percentile == 98.0
    assert stretch.low == pytest.approx(120.0)
    assert stretch.high == pytest.approx(1080.0)
    assert stretch.span == pytest.approx(960.0)


def test_outliers_are_clipped_not_allowed_to_dominate() -> None:
    """One saturated pixel must not compress everything else to black."""
    array = np.full((100, 100, 1), 1000.0, dtype=np.float32)
    array[0:50, :, 0] = 900.0
    array[0, 0, 0] = 1_000_000.0  # a single hot pixel

    result = normalize_to_uint8(array)

    assert result.array[0, 0, 0] == 255  # the outlier clips to white
    # The bulk of the image still spans the range instead of collapsing.
    assert result.array[10, 10, 0] == 0
    assert result.array[60, 60, 0] == 255


def test_percentiles_are_clipped_at_both_ends() -> None:
    array = np.arange(1000, dtype=np.float32).reshape(1000, 1, 1)
    result = normalize_to_uint8(array, lower_percentile=10.0, upper_percentile=90.0)

    assert result.array[:100, 0, 0].min() == 0  # below the 10th percentile
    assert result.array[900:, 0, 0].max() == 255  # above the 90th
    assert result.array[500, 0, 0] == pytest.approx(128, abs=2)


def test_per_band_stretches_independently() -> None:
    array = np.zeros((10, 10, 2), dtype=np.float32)
    array[:, :, 0] = np.linspace(0, 10, 100).reshape(10, 10)
    array[:, :, 1] = np.linspace(1000, 2000, 100).reshape(10, 10)

    per_band = normalize_to_uint8(array, per_band=True)
    assert per_band.bands[0].high < 20
    assert per_band.bands[1].high > 1000
    # Both bands reach full range when stretched independently.
    assert per_band.array[:, :, 0].max() == 255
    assert per_band.array[:, :, 1].max() == 255


def test_shared_stretch_preserves_relative_band_brightness() -> None:
    array = np.zeros((10, 10, 2), dtype=np.float32)
    array[:, :, 0] = np.linspace(0, 10, 100).reshape(10, 10)
    array[:, :, 1] = np.linspace(1000, 2000, 100).reshape(10, 10)

    shared = normalize_to_uint8(array, per_band=False)
    assert shared.bands[0] .low == shared.bands[1].low
    assert shared.bands[0].high == shared.bands[1].high
    # The dim band stays dim relative to the bright one.
    assert shared.array[:, :, 0].max() < shared.array[:, :, 1].min()


def test_flat_band_is_zeroed_rather_than_dividing_by_zero() -> None:
    array = np.full((16, 16, 1), 7.0, dtype=np.float32)
    result = normalize_to_uint8(array)

    assert result.bands[0].degenerate
    assert np.all(result.array == 0)


def test_near_flat_band_falls_back_to_full_range() -> None:
    """Percentiles can collapse onto one value even when the band is not flat."""
    array = np.zeros((100, 100, 1), dtype=np.float32)
    array[0, 0, 0] = 5.0  # 1 pixel in 10,000, far outside the 2-98 window

    result = normalize_to_uint8(array)

    assert not result.bands[0].degenerate
    assert result.bands[0].low == pytest.approx(0.0)
    assert result.bands[0].high == pytest.approx(5.0)
    assert result.array[0, 0, 0] == 255


def test_nodata_is_excluded_from_percentiles() -> None:
    array = np.full((100, 100, 1), -9999.0, dtype=np.float32)
    array[:10, :, 0] = np.linspace(10, 20, 1000).reshape(10, 100)

    result = normalize_to_uint8(array, nodata=-9999.0)

    assert result.bands[0].low > 0, "nodata must not drag the low end to -9999"
    assert result.bands[0].low == pytest.approx(10.2, abs=0.5)


def test_nan_is_excluded_from_percentiles() -> None:
    array = np.full((50, 50, 1), np.nan, dtype=np.float32)
    array[:10, :, 0] = np.linspace(100, 200, 500).reshape(10, 50)

    result = normalize_to_uint8(array)

    assert np.isfinite(result.bands[0].low)
    assert result.bands[0].low == pytest.approx(102, abs=3)


def test_non_finite_pixels_get_a_deterministic_output_value() -> None:
    """Casting NaN to uint8 is undefined behaviour; it must be handled first."""
    array = np.linspace(0, 100, 400, dtype=np.float32).reshape(20, 20, 1)
    array[0, 0, 0] = np.nan
    array[0, 1, 0] = np.inf
    array[0, 2, 0] = -np.inf

    result = normalize_to_uint8(array)

    assert result.array[0, 0, 0] == 0  # NaN  -> output minimum
    assert result.array[0, 1, 0] == 255  # +inf -> output maximum
    assert result.array[0, 2, 0] == 0  # -inf -> output minimum


def test_all_nodata_is_flagged_degenerate_not_guessed() -> None:
    array = np.full((8, 8, 1), -1.0, dtype=np.float32)
    result = normalize_to_uint8(array, nodata=-1.0)
    assert result.bands[0].degenerate


def test_stretch_is_reproducible_from_recorded_windows() -> None:
    """Recorded windows are enough to reproduce the result exactly."""
    rng = np.random.default_rng(7)
    array = (rng.random((32, 32, 3)) * 2000).astype(np.uint16)

    first = normalize_to_uint8(array)
    replayed = apply_stretch(array, first.bands, out_dtype=np.uint8)

    assert np.array_equal(first.array, replayed)


def test_invalid_percentiles_are_rejected() -> None:
    array = np.zeros((4, 4, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="lower < upper"):
        compute_band_stretches(array, lower_percentile=90.0, upper_percentile=10.0)
    with pytest.raises(ValueError, match="lower < upper"):
        compute_band_stretches(array, lower_percentile=-1.0, upper_percentile=50.0)


def test_rank_is_enforced() -> None:
    with pytest.raises(ValueError, match="rows, cols, bands"):
        compute_band_stretches(np.zeros((4, 4), dtype=np.float32))


def test_band_count_mismatch_is_rejected() -> None:
    array = np.zeros((4, 4, 2), dtype=np.float32)
    stretches = compute_band_stretches(array)
    with pytest.raises(ValueError, match="stretch windows"):
        apply_stretch(array, stretches[:1])


# ---------------------------------------------------------------------------
# On the real fixture
# ---------------------------------------------------------------------------


def test_normalises_the_synthetic_fixture(fixture_tif: Path) -> None:
    """The fixture's three grey levels must stay ordered and spread out."""
    scene = GeoTIFFLoader().load(fixture_tif)
    result = normalize_to_uint8(scene.array)

    assert result.array.shape == scene.array.shape
    assert result.array.dtype == np.uint8

    levels = np.unique(result.array)
    assert len(levels) == 3, "shadow / ground / roof must stay distinguishable"
    assert levels[0] == 0 and levels[-1] == 255
    # Ordering is preserved: shadow < ground < roof.
    assert levels[0] < levels[1] < levels[2]
