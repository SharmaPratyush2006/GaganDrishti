"""Tests for 512x512 tiling with 64 px overlap.

The important properties are that adjacent tiles really do overlap by the
configured amount, that every tile knows where it sits on the ground, and that
the recorded valid spans exactly partition the source -- which is what makes
future stitching possible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from depthwizard.config import AppConfig
from depthwizard.ingest.geotiff import pixel_to_map
from depthwizard.ingest.loaders import GeoTIFFLoader, ImageLoader
from depthwizard.ingest.synthetic import GeneratedFixture
from depthwizard.ingest.tiling import DEFAULT_OVERLAP, DEFAULT_TILE_SIZE, TileGrid, Tiler

LARGE_WIDTH_PX = 1200
LARGE_HEIGHT_PX = 900


# ---------------------------------------------------------------------------
# Configuration and defaults
# ---------------------------------------------------------------------------


def test_defaults_are_512_and_64() -> None:
    tiler = Tiler()
    assert tiler.tile_size == DEFAULT_TILE_SIZE == 512
    assert tiler.overlap == DEFAULT_OVERLAP == 64
    assert tiler.stride == 448


def test_config_carries_tiling_settings(app_config: AppConfig) -> None:
    tiling = app_config.ingest.tiling
    assert tiling.tile_size == 512
    assert tiling.overlap == 64
    assert tiling.stride == 448


def test_overlap_must_be_smaller_than_the_tile() -> None:
    with pytest.raises(ValueError, match="smaller than tile_size"):
        Tiler(tile_size=512, overlap=512)
    with pytest.raises(ValueError, match="must be >= 0"):
        Tiler(tile_size=512, overlap=-1)
    with pytest.raises(ValueError, match="must be positive"):
        Tiler(tile_size=0)


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------


def test_grid_layout_for_a_large_scene() -> None:
    grid = Tiler().build_grid(LARGE_HEIGHT_PX, LARGE_WIDTH_PX)

    # cols: offsets 0, 448, then the last pushed back to 1200-512=688
    # rows: offsets 0, then pushed back to 900-512=388
    assert (grid.n_tile_rows, grid.n_tile_cols) == (2, 3)
    assert len(grid) == 6
    assert sorted({t.col_off for t in grid}) == [0, 448, 688]
    assert sorted({t.row_off for t in grid}) == [0, 388]


def test_every_tile_is_full_size_when_the_source_is_large_enough() -> None:
    grid = Tiler().build_grid(LARGE_HEIGHT_PX, LARGE_WIDTH_PX)
    assert all(tile.height == 512 and tile.width == 512 for tile in grid)


def test_adjacent_tiles_overlap_by_the_configured_amount() -> None:
    grid = Tiler(tile_size=512, overlap=64).build_grid(LARGE_HEIGHT_PX, LARGE_WIDTH_PX)

    first_row = sorted([t for t in grid if t.tile_row == 0], key=lambda t: t.tile_col)
    a, b = first_row[0], first_row[1]
    overlap = (a.col_off + a.width) - b.col_off
    assert overlap == 64

    # The final column is pushed back from the edge, so it overlaps by MORE,
    # never less. Under-overlapping would leave an untreated seam.
    c = first_row[2]
    assert (b.col_off + b.width) - c.col_off >= 64


def test_tiles_cover_every_source_pixel() -> None:
    grid = Tiler().build_grid(LARGE_HEIGHT_PX, LARGE_WIDTH_PX)
    covered = np.zeros((LARGE_HEIGHT_PX, LARGE_WIDTH_PX), dtype=bool)
    for tile in grid:
        covered[tile.row_slice, tile.col_slice] = True
    assert covered.all(), "tiling left a gap"


def test_small_source_yields_one_short_tile() -> None:
    grid = Tiler().build_grid(100, 200)
    assert len(grid) == 1
    tile = grid[0]
    assert (tile.height, tile.width) == (100, 200)
    assert (tile.row_off, tile.col_off) == (0, 0)


def test_exactly_one_tile_when_source_equals_tile_size() -> None:
    grid = Tiler().build_grid(512, 512)
    assert len(grid) == 1
    assert grid[0].height == grid[0].width == 512


def test_zero_overlap_still_tiles_correctly() -> None:
    grid = Tiler(tile_size=256, overlap=0).build_grid(512, 512)
    assert len(grid) == 4
    assert {t.col_off for t in grid} == {0, 256}


def test_invalid_source_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Tiler().build_grid(0, 100)


# ---------------------------------------------------------------------------
# Valid spans: the stitching contract
# ---------------------------------------------------------------------------


def test_valid_spans_exactly_partition_the_source() -> None:
    """No gaps and no double-coverage: every pixel is owned by exactly one tile."""
    grid = Tiler().build_grid(LARGE_HEIGHT_PX, LARGE_WIDTH_PX)
    ownership = np.zeros((LARGE_HEIGHT_PX, LARGE_WIDTH_PX), dtype=np.int32)
    for tile in grid:
        ownership[tile.valid_row_slice, tile.valid_col_slice] += 1
    assert ownership.min() == 1 and ownership.max() == 1


def test_valid_span_lies_inside_its_own_tile() -> None:
    grid = Tiler().build_grid(LARGE_HEIGHT_PX, LARGE_WIDTH_PX)
    for tile in grid:
        assert tile.row_off <= tile.valid_row_start < tile.valid_row_end <= tile.row_off + tile.height
        assert tile.col_off <= tile.valid_col_start < tile.valid_col_end <= tile.col_off + tile.width


def test_local_valid_slices_index_the_tile_array() -> None:
    grid = Tiler().build_grid(LARGE_HEIGHT_PX, LARGE_WIDTH_PX)
    for tile in grid:
        local_rows = tile.local_valid_row_slice
        local_cols = tile.local_valid_col_slice
        assert 0 <= local_rows.start < local_rows.stop <= tile.height
        assert 0 <= local_cols.start < local_cols.stop <= tile.width


def test_source_can_be_reassembled_from_tile_valid_regions() -> None:
    """The end-to-end proof that the recorded bookkeeping is sufficient.

    Stitching is a later phase; this only demonstrates that Phase 1 stored
    enough information for it.
    """
    rng = np.random.default_rng(42)
    source = rng.integers(0, 256, size=(LARGE_HEIGHT_PX, LARGE_WIDTH_PX, 1), dtype=np.uint8)

    tiler = Tiler()
    grid = tiler.build_grid(LARGE_HEIGHT_PX, LARGE_WIDTH_PX)

    mosaic = np.zeros_like(source)
    for tile, pixels in tiler.iter_tiles(source, grid):
        core = pixels[tile.local_valid_row_slice, tile.local_valid_col_slice, :]
        mosaic[tile.valid_row_slice, tile.valid_col_slice, :] = core

    assert np.array_equal(mosaic, source)


# ---------------------------------------------------------------------------
# Tile-to-geographic mapping
# ---------------------------------------------------------------------------


def test_tiles_carry_geographic_transforms(large_fixture: GeneratedFixture) -> None:
    scene = GeoTIFFLoader().load(large_fixture.image_path)
    grid = Tiler().build_grid_for_scene(scene)

    assert grid.crs == scene.metadata.crs.to_string()
    assert all(tile.transform is not None for tile in grid)
    assert all(tile.bounds is not None for tile in grid)


def test_tile_origin_matches_the_parent_raster(large_fixture: GeneratedFixture) -> None:
    """A tile's map origin must equal the parent's coordinate at that pixel."""
    scene = GeoTIFFLoader().load(large_fixture.image_path)
    parent = scene.metadata.georeference.transform
    grid = Tiler().build_grid_for_scene(scene)

    for tile in grid:
        expected = pixel_to_map(parent, tile.col_off, tile.row_off)
        actual = pixel_to_map(tile.affine, 0, 0)
        assert actual == pytest.approx(expected)


def test_tiles_keep_the_parent_pixel_size(large_fixture: GeneratedFixture) -> None:
    scene = GeoTIFFLoader().load(large_fixture.image_path)
    gsd = scene.metadata.gsd_m
    for tile in Tiler().build_grid_for_scene(scene):
        assert tile.affine.a == pytest.approx(gsd)
        assert tile.affine.e == pytest.approx(-gsd)


def test_tile_bounds_are_consistent_with_its_transform(large_fixture: GeneratedFixture) -> None:
    scene = GeoTIFFLoader().load(large_fixture.image_path)
    for tile in Tiler().build_grid_for_scene(scene):
        west, south, east, north = tile.bounds
        assert east - west == pytest.approx(tile.width * scene.metadata.gsd_m)
        assert north - south == pytest.approx(tile.height * scene.metadata.gsd_m)


def test_tile_bounds_lie_within_the_parent_bounds(large_fixture: GeneratedFixture) -> None:
    scene = GeoTIFFLoader().load(large_fixture.image_path)
    p_west, p_south, p_east, p_north = scene.metadata.georeference.bounds
    for tile in Tiler().build_grid_for_scene(scene):
        west, south, east, north = tile.bounds
        assert p_west - 1e-6 <= west < east <= p_east + 1e-6
        assert p_south - 1e-6 <= south < north <= p_north + 1e-6


def test_relative_mode_scenes_tile_without_geography(png_path: Path) -> None:
    """A PNG has no map coordinates, but must still tile."""
    scene = ImageLoader().load(png_path)
    grid = Tiler().build_grid_for_scene(scene)

    assert len(grid) >= 1
    assert grid.crs is None
    assert all(tile.transform is None and tile.bounds is None for tile in grid)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extracted_tile_matches_the_source_window() -> None:
    rng = np.random.default_rng(1)
    source = rng.integers(0, 256, size=(700, 700, 2), dtype=np.uint8)
    tiler = Tiler()
    grid = tiler.build_grid(700, 700)

    for tile, pixels in tiler.iter_tiles(source, grid):
        assert pixels.shape == (tile.height, tile.width, 2)
        assert np.array_equal(pixels, source[tile.row_slice, tile.col_slice, :])


def test_mismatched_grid_and_array_is_rejected() -> None:
    tiler = Tiler()
    grid = tiler.build_grid(700, 700)
    wrong = np.zeros((600, 600, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="grid was planned for"):
        list(tiler.iter_tiles(wrong, grid))


# ---------------------------------------------------------------------------
# Persistence for future stitching
# ---------------------------------------------------------------------------


def test_grid_round_trips_through_json(large_fixture: GeneratedFixture, tmp_path: Path) -> None:
    scene = GeoTIFFLoader().load(large_fixture.image_path)
    grid = Tiler().build_grid_for_scene(scene)

    path = grid.save_json(tmp_path / "grid.json")
    restored = TileGrid.load_json(path)

    assert restored.to_dict() == grid.to_dict()
    assert len(restored) == len(grid)
    assert restored.crs == grid.crs
    assert restored[0].affine == grid[0].affine


def test_saved_grid_contains_what_stitching_needs(large_fixture: GeneratedFixture) -> None:
    scene = GeoTIFFLoader().load(large_fixture.image_path)
    payload = Tiler().build_grid_for_scene(scene).to_dict()

    for key in (
        "source_height", "source_width", "tile_size", "overlap", "stride",
        "n_tile_rows", "n_tile_cols", "crs", "source_transform", "tiles",
    ):
        assert key in payload

    tile = payload["tiles"][0]
    for key in (
        "index", "tile_row", "tile_col", "row_off", "col_off", "height", "width",
        "valid_row_start", "valid_row_end", "valid_col_start", "valid_col_end",
        "transform", "bounds",
    ):
        assert key in tile
