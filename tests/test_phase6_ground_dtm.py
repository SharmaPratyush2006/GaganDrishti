"""Phase 6: ground extraction, DTM interpolation, nDSM -- numerical behaviour.

Every raster is a small constructed SYNTHETIC surface: a planar terrain plus box
buildings of specified heights, so the terrain (DTM truth) and the object
heights (nDSM truth) are known exactly. A linear TIN reproduces a plane exactly,
so wherever the method is exact the tests use a float64 rounding tolerance
(1e-9 m), not a tuned one. Where it is not exact (nearest-ground extrapolation
at the border) the tolerance is the method's own bound, slope * distance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from depthwizard.surfaces.dtm import DtmError, compute_ndsm, interpolate_dtm
from depthwizard.surfaces.ground import GroundExtractionError, MorphologicalGroundExtractor, fill_invalid_nearest
from depthwizard.surfaces.phase6 import Phase6Error, derive_surfaces, gsd_from_transform
from depthwizard.surfaces.phase6_config import GroundFilterConfig

GSD = 0.5
EXACT = 1e-9


def surface(shape=(120, 120), *, base=100.0, slope_east=0.0, slope_north=0.0, buildings=(), nodata=None):
    """Planar terrain + box buildings ``(r0, r1, c0, c1, height_m)``; ``nodata`` = (r0, r1, c0, c1)."""
    rows, cols = np.indices(shape)
    terrain = base + slope_east * (cols + 0.5) * GSD - slope_north * (rows + 0.5) * GSD
    height = np.zeros(shape)
    for r0, r1, c0, c1, h in buildings:
        height[r0:r1, c0:c1] = h
    dsm = terrain + height
    valid = np.ones(shape, dtype=bool)
    if nodata is not None:
        r0, r1, c0, c1 = nodata
        dsm[r0:r1, c0:c1] = np.nan
        valid[r0:r1, c0:c1] = False
    return dsm, valid, terrain, height


def run(dsm, valid, extent_m, *, slope=0.05, min_height=2.0, iterations=5):
    cfg = GroundFilterConfig(max_terrain_slope=slope, min_object_height_m=min_height,
                             tin_refinement_max_iterations=iterations)
    return derive_surfaces(dsm, valid=valid, gsd_m=GSD, ground_cfg=cfg, max_building_extent_m=extent_m)


def assert_exact(result, valid, terrain, height):
    assert np.nanmax(np.abs(result["dtm"].dtm - terrain)[valid]) <= EXACT
    assert np.nanmax(np.abs(result["ndsm"].ndsm - height)[valid]) <= EXACT


# ---------------------------------------------------------------------------
# Ground extraction
# ---------------------------------------------------------------------------


def test_window_schedule_ends_strictly_above_the_largest_building():
    ex = MorphologicalGroundExtractor(max_building_extent_m=40.0, max_terrain_slope=0.05, min_object_height_m=2.0)
    assert ex.window_schedule(0.5) == [3, 5, 9, 17, 33, 65, 81]  # 81 px = 40.5 m > 40 m
    for extent in (0.4, 3.0, 10.25, 12.5, 40.0, 81.0):
        final = ex.__class__(max_building_extent_m=extent, max_terrain_slope=0.05,
                             min_object_height_m=2.0).window_schedule(0.5)[-1]
        assert final % 2 == 1 and final * 0.5 > extent
    assert ex.threshold_m(81, 0.5) == pytest.approx(2.0 + 0.05 * 40.5)


def test_extractor_requires_an_explicit_building_extent_and_positive_parameters():
    with pytest.raises(GroundExtractionError, match="required"):
        MorphologicalGroundExtractor(max_building_extent_m=None, max_terrain_slope=0.05, min_object_height_m=2.0)
    with pytest.raises(GroundExtractionError):
        MorphologicalGroundExtractor(max_building_extent_m=10, max_terrain_slope=0.0, min_object_height_m=2.0)
    with pytest.raises(GroundExtractionError, match="no valid pixel"):
        fill_invalid_nearest(np.zeros((3, 3)), np.zeros((3, 3), bool))


def test_ground_mask_is_the_complement_of_the_building():
    dsm, valid, terrain, height = surface(buildings=[(50, 70, 50, 70, 8.0)])
    ground = MorphologicalGroundExtractor(max_building_extent_m=10.0, max_terrain_slope=0.05,
                                          min_object_height_m=2.0).extract(dsm, valid=valid, gsd_m=GSD)
    np.testing.assert_array_equal(ground.ground, height == 0)
    assert ground.diagnostics["non_ground_px"] == 400


# ---------------------------------------------------------------------------
# Edge cases (DTM and nDSM)
# ---------------------------------------------------------------------------


def test_flat_terrain():
    dsm, valid, terrain, height = surface(buildings=[(50, 70, 50, 70, 8.0)])
    result = run(dsm, valid, 10.0)
    assert_exact(result, valid, terrain, height)
    assert np.all(result["dtm"].dtm[height == 0] == 100.0)  # ground pixels keep their DSM value exactly


def test_sloped_terrain_with_multiple_buildings():
    dsm, valid, terrain, height = surface(slope_east=0.02, slope_north=-0.015, buildings=[
        (10, 30, 10, 30, 30.0), (20, 35, 70, 100, 12.0), (60, 100, 20, 35, 45.0), (70, 95, 70, 95, 6.0)])
    result = run(dsm, valid, 20.0)
    assert_exact(result, valid, terrain, height)
    assert result["dtm"].diagnostics["extrapolated_px"] == 0
    for r0, r1, c0, c1, h in [(10, 30, 10, 30, 30.0), (70, 95, 70, 95, 6.0)]:
        # the nDSM reports height above ground, not the absolute elevation
        assert result["ndsm"].ndsm[(r0 + r1) // 2, (c0 + c1) // 2] == pytest.approx(h, abs=EXACT)
        assert dsm[(r0 + r1) // 2, (c0 + c1) // 2] > 100.0


def test_building_touching_the_raster_corner_is_extrapolated_within_its_bound():
    slope_east, slope_north = 0.02, -0.015
    dsm, valid, terrain, height = surface(slope_east=slope_east, slope_north=slope_north,
                                          buildings=[(0, 20, 0, 20, 10.0)])
    # Reflection at two borders makes the 20 px corner building 40 x 40 px: the window must exceed that.
    result = run(dsm, valid, 25.0)
    dtm = result["dtm"]
    assert not (dtm.ground & (height > 0)).any()
    assert dtm.extrapolated.any() and dtm.diagnostics["extrapolated_px"] == int(dtm.extrapolated.sum())
    bound = math.hypot(slope_east, slope_north) * GSD * dtm.distance_to_ground_px + EXACT
    error = np.abs(dtm.dtm - terrain)
    assert np.all(error[dtm.extrapolated] <= bound[dtm.extrapolated])
    assert np.all(np.abs(result["ndsm"].ndsm - height)[dtm.extrapolated] <= bound[dtm.extrapolated])
    assert np.max(error[~dtm.extrapolated]) <= EXACT


def test_nodata_region_stays_invalid_and_does_not_disturb_the_rest():
    dsm, valid, terrain, height = surface(slope_east=0.02, buildings=[(50, 70, 50, 70, 8.0)], nodata=(5, 25, 80, 110))
    result = run(dsm, valid, 10.0)
    assert np.isnan(result["dtm"].dtm[~valid]).all() and np.isnan(result["ndsm"].ndsm[~valid]).all()
    assert not result["dtm"].ground[~valid].any()
    assert_exact(result, valid, terrain, height)
    assert result["ndsm"].valid.sum() == valid.sum()


def test_all_ground_scene():
    dsm, valid, terrain, height = surface(slope_east=0.03, slope_north=0.01)
    result = run(dsm, valid, 10.0)
    assert result["dtm"].ground.all()
    np.testing.assert_array_equal(result["dtm"].dtm, dsm)
    assert np.all(result["ndsm"].ndsm == 0.0)
    assert result["dtm"].diagnostics["interpolated_px"] == 0


def test_small_building():
    dsm, valid, terrain, height = surface(buildings=[(60, 63, 60, 63, 5.0)])  # 3 x 3 px = 1.5 m
    result = run(dsm, valid, 1.5)
    assert_exact(result, valid, terrain, height)
    assert result["ground"].diagnostics["stages"][-1]["window_px"] == 5  # the 3 px window fits inside it


def test_large_building_region():
    dsm, valid, terrain, height = surface(slope_east=0.02, slope_north=-0.015, buildings=[(25, 95, 25, 95, 20.0)])
    result = run(dsm, valid, 35.0)  # 70 px = 35 m
    assert_exact(result, valid, terrain, height)
    assert result["dtm"].diagnostics["interpolated_px"] == 70 * 70


def test_building_larger_than_the_window_is_kept_as_ground():
    # Documented failure mode: max_building_extent_m below the true building size.
    dsm, valid, terrain, height = surface(buildings=[(40, 80, 40, 80, 15.0)])  # 40 px = 20 m
    result = run(dsm, valid, 10.0)
    assert result["dtm"].ground[60, 60]
    assert result["ndsm"].ndsm[60, 60] == 0.0  # the building is invisible in the nDSM


def test_dsm_equals_dtm_plus_ndsm_wherever_finite():
    dsm, valid, terrain, height = surface(slope_east=0.02, buildings=[(10, 30, 10, 30, 9.0), (70, 90, 60, 100, 4.0)],
                                          nodata=(100, 110, 0, 30))
    result = run(dsm, valid, 20.0)
    ok = result["ndsm"].valid
    assert ok.sum() == valid.sum()
    assert np.max(np.abs(result["dtm"].dtm[ok] + result["ndsm"].ndsm[ok] - dsm[ok])) <= EXACT


def test_negative_ndsm_values_are_preserved_and_counted():
    dsm = np.full((4, 4), 10.0)
    dtm = dsm.copy()
    dtm[1, 1] = 10.75
    dtm[2, 2] = np.nan
    result = compute_ndsm(dsm, dtm)
    assert result.ndsm[1, 1] == pytest.approx(-0.75)  # not clamped to 0
    assert result.negative_px == 1 and result.min_m == pytest.approx(-0.75)
    assert np.isnan(result.ndsm[2, 2]) and not result.valid[2, 2]
    assert result.summary()["negative_policy"] == "preserved, not clamped"


def test_tin_refinement_readmits_terrain_the_filter_misflagged():
    # A natural 8 m hill narrower than the window is raised terrain: the opening
    # flags its top as an object. The TIN refinement re-admits it ring by ring,
    # while the 8 m building (a vertical step) stays non-ground.
    dsm, valid, terrain, height = surface(buildings=[(90, 105, 90, 105, 8.0)])
    rows, cols = np.indices(dsm.shape)
    hill = 8.0 * np.exp(-((rows - 40) ** 2 + (cols - 40) ** 2) / (2 * 6.0 ** 2))
    dsm, terrain = dsm + hill, terrain + hill
    without = run(dsm, valid, 10.0, iterations=0)
    refined = run(dsm, valid, 10.0, iterations=5)
    assert ((height == 0) & ~without["dtm"].ground).sum() > 0
    assert np.max(np.abs(without["dtm"].dtm - terrain)) > 1.0  # the hilltop was cut off the DTM
    passes = refined["dtm"].diagnostics["tin_refinement"]["readmitted_per_pass"]
    assert sum(passes) > 0 and passes[-1] == 0  # stopped because nothing changed, not at the cap
    np.testing.assert_array_equal(refined["dtm"].ground, height == 0)
    assert_exact(refined, valid, terrain, height)


def test_dtm_needs_ground_and_degrades_explicitly_without_a_tin():
    dsm, valid, _, _ = surface(shape=(10, 10))
    with pytest.raises(DtmError, match="no valid pixel is ground"):
        interpolate_dtm(dsm, np.zeros_like(valid), valid)
    ground = np.zeros_like(valid)
    ground[0, 0] = ground[0, 1] = True  # two points cannot form a TIN
    result = interpolate_dtm(dsm, ground, valid)
    assert "no TIN" in result.diagnostics["tin_failure"]
    assert result.extrapolated.sum() == 98 and np.isfinite(result.dtm).all()
    with pytest.raises(DtmError, match="shape"):
        interpolate_dtm(dsm, ground[:5], valid)


def test_gsd_requires_square_north_up_pixels():
    from rasterio.transform import Affine

    assert gsd_from_transform(Affine(0.5, 0, 0, 0, -0.5, 0)) == 0.5
    with pytest.raises(Phase6Error, match="square"):
        gsd_from_transform(Affine(0.5, 0, 0, 0, -0.25, 0))
    with pytest.raises(Phase6Error, match="rotated"):
        gsd_from_transform(Affine(0.5, 0.1, 0, 0, -0.5, 0))
