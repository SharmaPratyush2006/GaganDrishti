"""Phase 4b: DEM onto the image grid, DSM = T + a*exp(z_rel) + b, COG export.

All inputs are SYNTHETIC. The terrain is a tilted plane, which bilinear
resampling reproduces exactly, so the georeferencing tolerance is set by float32
storage (~3e-5 m at 540 m), not interpolation; a half-pixel georeferencing
error would show as slope * 0.25 m = 5 mm (see GEOMETRIC_TOLERANCE_M).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from depthwizard.calibration.terrain import GridSpec, TerrainError, load_terrain_on_grid
from depthwizard.ingest.geotiff import CogError, validate_cog, write_cog, write_single_band
from depthwizard.ingest.synthetic import TerrainSpec, render_scene, terrain_truth_on_grid, write_synthetic_dem
from depthwizard.surfaces.dsm import DsmError, fuse_dsm
from depthwizard.surfaces.run import DSM_NODATA, GEOMETRIC_TOLERANCE_M

UTM = "EPSG:32643"
GRID = GridSpec(crs=UTM, transform=Affine(0.5, 0.0, 700000.0, 0.0, -0.5, 3170000.0), width=64, height=48)


def _plane_dem(tmp_path, *, crs=UTM, transform=Affine(10.0, 0.0, 699900.0, 0.0, -10.0, 3170100.0),
               shape=(40, 40), nodata=-32768.0, name="dem.tif"):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    easting = transform.c + (cols + 0.5) * transform.a
    northing = transform.f + (rows + 0.5) * transform.e
    values = (500.0 + 0.02 * (easting - 700000.0) - 0.01 * (northing - 3170000.0)).astype(np.float64)
    return write_single_band(tmp_path / name, values, crs=crs, transform=transform, nodata=nodata), values


def _plane_truth(grid: GridSpec) -> np.ndarray:
    rows, cols = np.mgrid[0:grid.height, 0:grid.width]
    easting = grid.transform.c + (cols + 0.5) * grid.transform.a
    northing = grid.transform.f + (rows + 0.5) * grid.transform.e
    return 500.0 + 0.02 * (easting - 700000.0) - 0.01 * (northing - 3170000.0)


# --- A-C, missing transform: fail loudly ------------------------------------------


def test_dem_without_crs_fails(tmp_path) -> None:
    path = tmp_path / "no_crs.tif"
    with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=1, dtype="float32",
                       transform=Affine(10, 0, 699900, 0, -10, 3170100)) as dst:
        dst.write(np.ones((8, 8), np.float32), 1)
    with pytest.raises(TerrainError, match="DEM .* has no CRS"):
        load_terrain_on_grid(path, GRID)


def test_image_without_crs_fails(tmp_path) -> None:
    dem, _ = _plane_dem(tmp_path)
    with pytest.raises(TerrainError, match="image grid has no CRS"):
        load_terrain_on_grid(dem, GridSpec(crs=None, transform=GRID.transform, width=64, height=48))


def test_missing_transform_fails(tmp_path) -> None:
    dem, _ = _plane_dem(tmp_path, transform=Affine.identity(), name="identity.tif")
    with pytest.raises(TerrainError, match="no spatial transform"):
        load_terrain_on_grid(dem, GRID)
    good, _ = _plane_dem(tmp_path)
    with pytest.raises(TerrainError, match="no spatial transform"):
        load_terrain_on_grid(good, GridSpec(crs=UTM, transform=Affine.identity(), width=64, height=48))


def test_non_overlapping_dem_fails(tmp_path) -> None:
    far = Affine(10.0, 0.0, 800000.0, 0.0, -10.0, 3170100.0)
    dem, _ = _plane_dem(tmp_path, transform=far)
    with pytest.raises(TerrainError, match="does not overlap"):
        load_terrain_on_grid(dem, GRID)


def test_invalid_dimensions_fail(tmp_path) -> None:
    dem, _ = _plane_dem(tmp_path)
    with pytest.raises(TerrainError, match="invalid dimensions"):
        load_terrain_on_grid(dem, GridSpec(crs=UTM, transform=GRID.transform, width=0, height=48))


# --- D-E: reprojection and grid alignment -----------------------------------------


def test_dem_in_other_crs_is_reprojected_onto_the_image_grid(tmp_path, scene) -> None:
    dem = write_synthetic_dem(scene, tmp_path / "dem_4326.tif", TerrainSpec())
    raster = scene.raster
    grid = GridSpec(crs=raster.crs, transform=Affine(raster.gsd_m, 0, raster.origin_easting_m,
                                                     0, -raster.gsd_m, raster.origin_northing_m),
                    width=raster.width_px, height=raster.height_px)
    placed = load_terrain_on_grid(dem.path, grid)
    assert placed.provenance["source_crs"] == "EPSG:4326"
    assert placed.provenance["target_crs"] == raster.crs
    assert placed.provenance["reprojected"] and placed.provenance["resampling"] == "bilinear"
    assert placed.values.shape == (raster.height_px, raster.width_px)
    assert placed.provenance["target_transform"] == [float(v) for v in tuple(grid.transform)[:6]]
    assert placed.valid.all()
    # J (terrain part): matches the analytic terrain at pixel centres.
    truth = terrain_truth_on_grid(scene, TerrainSpec())
    assert np.abs(placed.values - truth).max() <= GEOMETRIC_TOLERANCE_M


def test_same_crs_coarser_dem_aligns_exactly(tmp_path) -> None:
    dem, _ = _plane_dem(tmp_path)  # 10 m pixels vs the 0.5 m image grid
    placed = load_terrain_on_grid(dem, GRID)
    assert placed.values.shape == (GRID.height, GRID.width)
    assert not placed.provenance["reprojected"]
    np.testing.assert_allclose(placed.values, _plane_truth(GRID), atol=1e-6)


def test_dem_nodata_is_masked_not_zero_filled(tmp_path) -> None:
    transform = Affine(10.0, 0.0, 699900.0, 0.0, -10.0, 3170100.0)
    path, values = _plane_dem(tmp_path)
    # DEM columns < 12 (E < 700020) are nodata; the image spans E 700000-700032.
    holed = values.copy()
    holed[:, :12] = -32768.0
    write_single_band(path, holed, crs=UTM, transform=transform, nodata=-32768.0)
    placed = load_terrain_on_grid(path, GRID)
    assert not placed.valid[:, :40].any()  # image columns 0-39 lie at E < 700020
    assert placed.valid.any()
    assert np.isnan(placed.values[~placed.valid]).all()
    np.testing.assert_allclose(placed.values[placed.valid], _plane_truth(GRID)[placed.valid], atol=1e-6)

    # A hole covering the whole image grid: loud failure, not a zero-filled grid.
    holed[:, :17] = -32768.0
    write_single_band(path, holed, crs=UTM, transform=transform, nodata=-32768.0)
    with pytest.raises(TerrainError, match="no valid terrain pixel"):
        load_terrain_on_grid(path, GRID)


# --- F-G: the DSM equation and nodata ----------------------------------------------


def test_dsm_equation() -> None:
    rng = np.random.default_rng(3)
    terrain = rng.uniform(-20.0, 900.0, (6, 7))
    z = rng.uniform(-1.0, 3.0, (6, 7)).astype(np.float32)
    a, b = 0.61, -1.0
    result = fuse_dsm(terrain, np.ones_like(terrain, bool), z, a, b)
    np.testing.assert_allclose(result.dsm, terrain + a * np.exp(z.astype(np.float64)) + b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.agl, a * np.exp(z.astype(np.float64)) + b, rtol=0, atol=1e-12)


def test_nodata_propagates_and_zero_is_valid(tmp_path) -> None:
    terrain = np.zeros((4, 4))           # 0 m is a real elevation, not nodata
    t_valid = np.ones((4, 4), bool)
    t_valid[0, 0] = False
    z = np.zeros((4, 4))
    z[1, 1] = np.nan
    result = fuse_dsm(terrain, t_valid, z, 1.0, 0.0)
    assert np.isnan(result.dsm[0, 0]) and np.isnan(result.dsm[1, 1])
    assert result.valid.sum() == 14 and np.all(result.dsm[result.valid] == 1.0)
    path = write_cog(tmp_path / "dsm.tif", result.dsm.astype(np.float32), crs=UTM,
                     transform=GRID.transform, nodata=DSM_NODATA)
    with rasterio.open(path) as src:
        data = src.read(1)
        assert src.nodata == DSM_NODATA
    assert data[0, 0] == DSM_NODATA and data[1, 1] == DSM_NODATA and data[2, 2] == 1.0


@pytest.mark.parametrize("a, b, message", [
    (math.nan, 0.0, "finite"), (1.0, math.inf, "finite"), (0.0, 0.0, "not positive"), (-0.5, 1.0, "not positive"),
])
def test_invalid_calibration_parameters_fail(a, b, message) -> None:
    with pytest.raises(DsmError, match=message):
        fuse_dsm(np.zeros((2, 2)), np.ones((2, 2), bool), np.zeros((2, 2)), a, b)


def test_entirely_invalid_grids_fail() -> None:
    with pytest.raises(DsmError, match="terrain grid is entirely invalid"):
        fuse_dsm(np.full((2, 2), np.nan), np.ones((2, 2), bool), np.zeros((2, 2)), 1.0, 0.0)
    with pytest.raises(DsmError, match="no valid AGL"):
        fuse_dsm(np.zeros((2, 2)), np.ones((2, 2), bool), np.full((2, 2), np.nan), 1.0, 0.0)


# --- H-I: COG metadata and readability ---------------------------------------------


def test_cog_metadata_and_readability(tmp_path) -> None:
    grid = GridSpec(crs=UTM, transform=Affine(0.5, 0, 700000.0, 0, -0.5, 3170000.0), width=600, height=520)
    data = np.linspace(400.0, 700.0, grid.width * grid.height, dtype=np.float32).reshape(grid.height, grid.width)
    data[:5, :5] = np.nan
    path = write_cog(tmp_path / "x.tif", data, crs=UTM, transform=grid.transform, nodata=DSM_NODATA)
    record = validate_cog(path, crs=UTM, transform=grid.transform, width=grid.width, height=grid.height,
                          nodata=DSM_NODATA, dtype="float32", expected=data.astype(np.float64))
    assert record["ok"] and record["layout"] == "COG" and record["tiled"]
    assert record["overview_factors"], "a raster larger than one tile must have overviews"
    assert record["nodata_pixels"] == 25
    assert record["bounds"] == [700000.0, 3170000.0 - 260.0, 700300.0, 3170000.0]

    with pytest.raises(CogError, match="CRS"):
        validate_cog(path, crs="EPSG:4326", transform=grid.transform, width=grid.width,
                     height=grid.height, nodata=DSM_NODATA, dtype="float32")
    with pytest.raises(CogError, match="dimensions"):
        validate_cog(path, crs=UTM, transform=grid.transform, width=grid.width + 1,
                     height=grid.height, nodata=DSM_NODATA, dtype="float32")


def test_cog_refuses_colliding_nodata(tmp_path) -> None:
    with pytest.raises(CogError, match="collides"):
        write_cog(tmp_path / "bad.tif", np.full((4, 4), DSM_NODATA, np.float32), crs=UTM,
                  transform=GRID.transform, nodata=DSM_NODATA)


# --- J: SYNTHETIC end-to-end accuracy -----------------------------------------------


def test_synthetic_dsm_matches_truth(tmp_path, scene, fixture_tif) -> None:
    grid = GridSpec.from_raster(fixture_tif)
    dem = write_synthetic_dem(scene, tmp_path / "dem.tif", TerrainSpec())
    placed = load_terrain_on_grid(dem.path, grid)
    agl_true = render_scene(scene).height_m.astype(np.float64)
    c = 0.5
    z_rel = (np.log(agl_true + 1.0) + c).astype(np.float32)
    result = fuse_dsm(placed.values, placed.valid, z_rel, math.exp(-c), -1.0)
    truth = terrain_truth_on_grid(scene, TerrainSpec()) + agl_true
    assert result.valid.all()
    assert np.abs(result.dsm - truth).max() <= GEOMETRIC_TOLERANCE_M
