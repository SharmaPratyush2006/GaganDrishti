"""Pair discovery, spatial splitting, crop geometry and masking.

Runs on the synthetic DFC2019-*shaped* fixtures from ``conftest`` -- see the note
there: those are arbitrary synthetic rasters that share DFC2019's layout, not
DFC2019 data. Nothing in this file asserts an accuracy, because nothing here
predicts anything.

The invariant under test throughout is the one the module promises: **one crop
window, applied to both rasters**, with invalid pixels excluded rather than
imputed.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from depthwizard.relative.config import DatasetConfig, SplitConfig
from depthwizard.relative.data import (
    CropWindow,
    DatasetError,
    ScenePair,
    check_pair_registration,
    denormalize_image,
    discover_pairs,
    expected_steps,
    normalize_image,
    plan_train_crops,
    plan_validation_crops,
    raster_shape,
    read_pair_tile,
    sample_train_tile,
    split_pairs,
    summarize_pairs,
)

from conftest import (
    DFC_HEIGHT_SUBDIR,
    DFC_HEIGHT_SUFFIX,
    DFC_IMAGE_SUBDIR,
    DFC_NODATA,
    write_synthetic_pair,
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovers_every_pair(dataset_config):
    pairs = discover_pairs(dataset_config)
    assert len(pairs) == 4
    assert {p.scene_id for p in pairs} == {"JAX_004", "OMA_012"}
    assert all(p.image_path.is_file() and p.height_path.is_file() for p in pairs)


def test_pairs_are_sorted_so_discovery_is_reproducible(dataset_config):
    stems = [p.stem for p in discover_pairs(dataset_config)]
    assert stems == sorted(stems)


def test_missing_root_names_the_path_and_says_no_data_ships(tmp_path):
    cfg = DatasetConfig(
        root=tmp_path / "absent",
        split=SplitConfig(mode="explicit", val_scene_ids=("x",)),
    )
    with pytest.raises(DatasetError, match="does not exist") as excinfo:
        discover_pairs(cfg)
    assert "ships no DFC2019 data" in str(excinfo.value)


def test_missing_subdirectory_is_reported(dataset_config, dfc_like_root):
    import shutil

    shutil.rmtree(dfc_like_root / DFC_HEIGHT_SUBDIR)
    with pytest.raises(DatasetError, match="height_subdir"):
        discover_pairs(dataset_config)


def test_no_matching_images_is_reported(dataset_config):
    cfg = DatasetConfig(
        root=dataset_config.root,
        image_suffix="_NOSUCH.tif",
        split=dataset_config.split,
    )
    with pytest.raises(DatasetError, match="no files ending in"):
        discover_pairs(cfg)


def test_an_image_without_its_height_file_is_an_error(dataset_config, dfc_like_root):
    """A half-present dataset must fail, not train on whatever is there."""
    (dfc_like_root / DFC_HEIGHT_SUBDIR / f"JAX_004_001{DFC_HEIGHT_SUFFIX}").unlink()
    with pytest.raises(DatasetError, match="no matching height file"):
        discover_pairs(dataset_config)


def test_unmatched_regex_falls_back_to_the_stem(dfc_like_root):
    """Each unmatched file becomes its own scene: weaker, but never merging areas."""
    cfg = DatasetConfig(
        root=dfc_like_root,
        scene_id_regex=r"^(?P<scene>ZZZ_\d+)",
        split=SplitConfig(mode="explicit", val_scene_ids=("JAX_004_001",)),
    )
    pairs = discover_pairs(cfg)
    assert {p.scene_id for p in pairs} == {p.stem for p in pairs}


def test_summarize_pairs_counts_scenes_not_tiles(dataset_config):
    summary = summarize_pairs(discover_pairs(dataset_config))
    assert summary["pairs"] == 4
    assert summary["scenes"] == 2
    assert summary["scene_ids"] == ["JAX_004", "OMA_012"]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_prefix_split_holds_out_whole_scenes(dataset_config):
    pairs = discover_pairs(dataset_config)
    train, val = split_pairs(pairs, SplitConfig(mode="scene_prefix",
                                                val_scene_prefixes=("OMA",)))
    assert {p.scene_id for p in train} == {"JAX_004"}
    assert {p.scene_id for p in val} == {"OMA_012"}
    # The two sides share no scene at all -- that is what "spatial" means here.
    assert not {p.scene_id for p in train} & {p.scene_id for p in val}


def test_explicit_split_holds_out_the_listed_ids(dataset_config):
    pairs = discover_pairs(dataset_config)
    train, val = split_pairs(pairs, SplitConfig(mode="explicit",
                                                val_scene_ids=("JAX_004",)))
    assert {p.scene_id for p in val} == {"JAX_004"}
    assert {p.scene_id for p in train} == {"OMA_012"}


def test_random_scene_split_is_seeded(dataset_config):
    pairs = discover_pairs(dataset_config)
    split = SplitConfig(mode="random_scene", val_fraction=0.5, seed=7)
    first = split_pairs(pairs, split)
    second = split_pairs(pairs, split)
    assert [p.stem for p in first[0]] == [p.stem for p in second[0]]
    assert [p.stem for p in first[1]] == [p.stem for p in second[1]]


def test_random_tile_split_returns_everything_on_both_sides(dataset_config):
    """It is not a spatial split, and the code says so rather than pretending."""
    pairs = discover_pairs(dataset_config)
    train, val = split_pairs(pairs, SplitConfig(mode="random_tile"))
    assert len(train) == len(val) == len(pairs)


def test_split_that_empties_training_is_an_error(dataset_config):
    pairs = discover_pairs(dataset_config)
    with pytest.raises(DatasetError, match="training set empty"):
        split_pairs(pairs, SplitConfig(mode="scene_prefix",
                                       val_scene_prefixes=("JAX", "OMA")))


def test_split_with_no_validation_warns_but_proceeds(dataset_config, caplog):
    pairs = discover_pairs(dataset_config)
    train, val = split_pairs(pairs, SplitConfig(mode="scene_prefix",
                                                val_scene_prefixes=("NOPE",)))
    assert len(train) == 4 and val == ()


def test_splitting_nothing_is_an_error():
    with pytest.raises(DatasetError, match="empty pair list"):
        split_pairs((), SplitConfig(mode="explicit", val_scene_ids=("x",)))


# ---------------------------------------------------------------------------
# Crop geometry
# ---------------------------------------------------------------------------


def test_validation_crops_tile_the_scene_without_gaps(dataset_config):
    windows = plan_validation_crops((96, 96), dataset_config)  # tile 32
    assert len(windows) == 9
    covered = np.zeros((96, 96), dtype=int)
    for window in windows:
        covered[window.row_slice, window.col_slice] += 1
    assert covered.min() == 1 and covered.max() == 1  # exact partition


def test_validation_crops_are_deterministic(dataset_config):
    assert plan_validation_crops((96, 96), dataset_config) == plan_validation_crops(
        (96, 96), dataset_config
    )


def test_last_validation_crop_is_pushed_back_not_padded(dataset_config):
    """A 100 px extent at tile 32 ends with a window starting at 68, not 96."""
    windows = plan_validation_crops((100, 32), dataset_config)
    row_offsets = sorted({w.row_off for w in windows})
    assert row_offsets == [0, 32, 64, 68]
    assert all(w.row_off + w.size <= 100 for w in windows)


def test_scene_smaller_than_a_tile_yields_one_window(dataset_config):
    assert plan_validation_crops((20, 20), dataset_config) == (
        CropWindow(row_off=0, col_off=0, size=32),
    )


def test_train_crops_stay_inside_the_raster(dataset_config):
    pairs = discover_pairs(dataset_config)
    rng = random.Random(0)
    for _ in range(50):
        window = plan_train_crops(pairs[0], (96, 96), dataset_config, rng)
        assert 0 <= window.row_off <= 96 - 32
        assert 0 <= window.col_off <= 96 - 32


def test_train_crops_are_reproducible_from_the_seed(dataset_config):
    pairs = discover_pairs(dataset_config)
    draw = lambda: [  # noqa: E731 - a one-line helper reads better here
        plan_train_crops(pairs[0], (96, 96), dataset_config, random.Random(42))
        for _ in range(5)
    ]
    assert draw() == draw()


def test_crop_window_slices_have_the_requested_size():
    window = CropWindow(row_off=10, col_off=20, size=32)
    assert window.row_slice == slice(10, 42)
    assert window.col_slice == slice(20, 52)


# ---------------------------------------------------------------------------
# Paired reading
# ---------------------------------------------------------------------------


def test_read_pair_tile_returns_aligned_shapes(dataset_config):
    pair = discover_pairs(dataset_config)[0]
    sample = read_pair_tile(pair, CropWindow(0, 0, 32), dataset_config)
    assert sample.image.shape == (3, 32, 32)
    assert sample.height.shape == (32, 32)
    assert sample.valid.shape == (32, 32)
    assert sample.image.dtype == np.float32
    assert sample.valid.dtype == np.bool_


def test_nodata_rows_are_masked_out_not_imputed(dataset_config):
    """The fixture's first 8 rows are the sentinel; they must be invalid."""
    pair = discover_pairs(dataset_config)[0]
    sample = read_pair_tile(pair, CropWindow(0, 0, 32), dataset_config)
    assert not sample.valid[:8, :].any()
    assert sample.valid[8:, :].all()
    # Masked pixels are zeroed, never left holding the sentinel.
    assert not np.isclose(sample.height, DFC_NODATA).any()
    assert np.isfinite(sample.height).all()


def test_valid_fraction_matches_the_mask(dataset_config):
    pair = discover_pairs(dataset_config)[0]
    sample = read_pair_tile(pair, CropWindow(0, 0, 32), dataset_config)
    assert sample.valid_fraction == pytest.approx(24 / 32)


def test_padding_outside_the_raster_is_marked_invalid(dataset_config):
    """A window hanging off the edge is padded, and the padding is not a measurement."""
    pair = discover_pairs(dataset_config)[0]
    sample = read_pair_tile(pair, CropWindow(80, 80, 32), dataset_config)
    # Rows/cols 16.. of the window are past the 96 px raster.
    assert sample.valid[16:, :].sum() == 0
    assert sample.valid[:, 16:].sum() == 0
    assert np.isfinite(sample.height).all()


def test_metadata_records_the_window_and_the_scene(dataset_config):
    pair = discover_pairs(dataset_config)[0]
    sample = read_pair_tile(pair, CropWindow(32, 0, 32), dataset_config)
    meta = sample.metadata
    assert meta["stem"] == pair.stem
    assert meta["scene_id"] == pair.scene_id
    assert meta["row_off"] == 32 and meta["col_off"] == 0
    assert meta["tile_size"] == 32
    assert meta["height_resampled"] is False
    # The dfc_like_root fixture is georeferenced (the real release is not; see
    # the no-CRS tests below).
    assert meta["crs"].startswith("EPSG:")
    assert meta["georeferenced"] is True
    assert len(meta["transform"]) == 6


def test_size_mismatch_is_refused_by_default(tmp_path):
    root = tmp_path / "mismatch"
    write_synthetic_pair(root, "JAX_001_001", size=64, height_size=32)
    cfg = DatasetConfig(
        root=root,
        tile_size=32,
        split=SplitConfig(mode="explicit", val_scene_ids=("x",)),
    )
    pair = discover_pairs(cfg)[0]
    with pytest.raises(DatasetError, match="Refusing to guess"):
        read_pair_tile(pair, CropWindow(0, 0, 32), cfg)


def test_size_mismatch_resample_is_explicit_and_recorded(tmp_path):
    root = tmp_path / "mismatch_ok"
    write_synthetic_pair(root, "JAX_001_001", size=64, height_size=32)
    cfg = DatasetConfig(
        root=root,
        tile_size=32,
        on_size_mismatch="resample_height_to_image",
        split=SplitConfig(mode="explicit", val_scene_ids=("x",)),
    )
    pair = discover_pairs(cfg)[0]
    sample = read_pair_tile(pair, CropWindow(0, 0, 32), cfg)
    assert sample.height.shape == (32, 32)
    assert sample.metadata["height_resampled"] is True


def test_scene_smaller_than_a_tile_is_refused_when_padding_is_off(tmp_path):
    root = tmp_path / "tiny"
    write_synthetic_pair(root, "JAX_001_001", size=16)
    cfg = DatasetConfig(
        root=root,
        tile_size=32,
        allow_padding=False,
        split=SplitConfig(mode="explicit", val_scene_ids=("x",)),
    )
    pair = discover_pairs(cfg)[0]
    with pytest.raises(DatasetError, match="allow_padding"):
        read_pair_tile(pair, CropWindow(0, 0, 32), cfg)


def test_sample_train_tile_returns_its_real_mask_even_when_retries_fail(tmp_path):
    """A crop that never meets min_valid_fraction is kept, mask and all.

    Dropping it would bias the epoch towards dense areas; keeping it is safe
    because the loss only ever sees valid pixels.
    """
    root = tmp_path / "mostly_nodata"
    write_synthetic_pair(root, "JAX_001_001", size=64, nodata_rows=64)
    cfg = DatasetConfig(
        root=root,
        tile_size=32,
        min_valid_fraction=0.99,
        max_crop_attempts=3,
        split=SplitConfig(mode="explicit", val_scene_ids=("x",)),
    )
    pair = discover_pairs(cfg)[0]
    sample = sample_train_tile(pair, (64, 64), cfg, random.Random(0))
    assert sample.valid_fraction == 0.0
    assert np.isfinite(sample.height).all()


def test_heights_above_valid_max_are_masked(tmp_path):
    root = tmp_path / "toohigh"
    write_synthetic_pair(root, "JAX_001_001", size=32, fill_height=5000.0)
    cfg = DatasetConfig(
        root=root,
        tile_size=32,
        height_valid_max=1000.0,
        split=SplitConfig(mode="explicit", val_scene_ids=("x",)),
    )
    pair = discover_pairs(cfg)[0]
    sample = read_pair_tile(pair, CropWindow(0, 0, 32), cfg)
    assert sample.valid.sum() == 0


# ---------------------------------------------------------------------------
# Spatial registration: CRS and transform, not just pixel dimensions
# ---------------------------------------------------------------------------


def _single_pair(root, **pair_kwargs):
    write_synthetic_pair(root, "JAX_001_001", **{"size": 64, **pair_kwargs})
    cfg = DatasetConfig(
        root=root,
        tile_size=32,
        on_size_mismatch="resample_height_to_image",
        split=SplitConfig(mode="explicit", val_scene_ids=("x",)),
    )
    return discover_pairs(cfg)[0], cfg


def _rewrite_height(pair, *, transform=None, crs="keep"):
    """Rewrite the pair's height raster with a different georeferencing."""
    import rasterio

    with rasterio.open(pair.height_path) as src:
        data = src.read(1)
        profile = src.profile
    if transform is not None:
        profile["transform"] = transform
    if crs != "keep":
        profile["crs"] = crs
    with rasterio.open(pair.height_path, "w", **profile) as dst:
        dst.write(data, 1)


def _shifted(transform, d_col=0.0, d_row=0.0, pixel=None):
    from rasterio.transform import Affine

    a = transform.a if pixel is None else pixel
    e = transform.e if pixel is None else -pixel
    return Affine(a, transform.b, transform.c + d_col * transform.a,
                  transform.d, e, transform.f + d_row * transform.e)


def _image_transform(pair):
    import rasterio

    with rasterio.open(pair.image_path) as src:
        return src.transform


def test_aligned_pair_passes_registration(tmp_path):
    pair, cfg = _single_pair(tmp_path / "ok")
    check_pair_registration(pair)
    assert read_pair_tile(pair, CropWindow(0, 0, 32), cfg).valid.any()


def test_same_size_pair_offset_by_pixels_is_refused(tmp_path):
    """Equal dimensions, 2 px apart on the ground: the old check let this through."""
    pair, cfg = _single_pair(tmp_path / "offset")
    _rewrite_height(pair, transform=_shifted(_image_transform(pair), d_col=2.0))
    with pytest.raises(DatasetError, match="misregistered"):
        read_pair_tile(pair, CropWindow(0, 0, 32), cfg)
    with pytest.raises(DatasetError, match="misregistered"):
        check_pair_registration(pair)


def test_same_size_pair_with_a_different_pixel_size_is_refused(tmp_path):
    pair, cfg = _single_pair(tmp_path / "gsd")
    _rewrite_height(pair, transform=_shifted(_image_transform(pair), pixel=0.6))
    with pytest.raises(DatasetError, match="misregistered"):
        read_pair_tile(pair, CropWindow(0, 0, 32), cfg)


def test_different_crs_is_refused(tmp_path):
    pair, cfg = _single_pair(tmp_path / "crs")
    _rewrite_height(pair, crs="EPSG:32644")
    with pytest.raises(DatasetError, match="CRS"):
        read_pair_tile(pair, CropWindow(0, 0, 32), cfg)


def test_height_without_a_crs_is_refused(tmp_path):
    pair, cfg = _single_pair(tmp_path / "nocrs")
    _rewrite_height(pair, crs=None)
    with pytest.raises(DatasetError, match="only the image raster has a CRS"):
        read_pair_tile(pair, CropWindow(0, 0, 32), cfg)


def test_rotated_height_grid_is_refused(tmp_path):
    from rasterio.transform import Affine

    pair, cfg = _single_pair(tmp_path / "rot")
    t = _image_transform(pair)
    _rewrite_height(pair, transform=Affine(t.a, 0.05, t.c, 0.05, t.e, t.f))
    with pytest.raises(DatasetError, match="rotation"):
        read_pair_tile(pair, CropWindow(0, 0, 32), cfg)


def test_sub_pixel_float_jitter_is_tolerated(tmp_path):
    """GeoTIFF tags carry float round-off; 1/10000 px is not misregistration."""
    pair, cfg = _single_pair(tmp_path / "jitter")
    _rewrite_height(pair, transform=_shifted(_image_transform(pair), d_col=1e-4))
    check_pair_registration(pair)


def test_resampled_pair_must_still_cover_the_same_ground(tmp_path):
    """on_size_mismatch resamples grids; it must not excuse a shifted footprint."""
    pair, cfg = _single_pair(tmp_path / "resample_ok", height_size=32)
    read_pair_tile(pair, CropWindow(0, 0, 32), cfg)  # same footprint: fine

    pair, cfg = _single_pair(tmp_path / "resample_bad", height_size=32)
    import rasterio

    with rasterio.open(pair.height_path) as src:
        shifted = _shifted(src.transform, d_col=3.0)
    _rewrite_height(pair, transform=shifted)
    with pytest.raises(DatasetError, match="misregistered"):
        read_pair_tile(pair, CropWindow(0, 0, 32), cfg)


def test_misregistered_scene_fails_when_the_dataset_is_built(tmp_path):
    pytest.importorskip("torch", reason="the Dataset layer needs torch")
    from depthwizard.relative.dataset import RelativeHeightDataset

    pair, cfg = _single_pair(tmp_path / "early")
    _rewrite_height(pair, transform=_shifted(_image_transform(pair), d_row=4.0))
    with pytest.raises(DatasetError, match="misregistered"):
        RelativeHeightDataset([pair], cfg, train=True)


def test_valid_negative_heights_are_kept_as_measurements(tmp_path):
    """A slightly negative AGL value is real data, not nodata: kept, unaltered."""
    root = tmp_path / "negative"
    write_synthetic_pair(root, "JAX_001_001", size=32, fill_height=-2.5, nodata_rows=4)
    cfg = DatasetConfig(
        root=root,
        tile_size=32,
        split=SplitConfig(mode="explicit", val_scene_ids=("x",)),
    )
    pair = discover_pairs(cfg)[0]
    sample = read_pair_tile(pair, CropWindow(0, 0, 32), cfg)
    assert sample.valid[4:, :].all()
    assert np.allclose(sample.height[4:, :], -2.5)
    assert not sample.valid[:4, :].any()  # the sentinel rows are still masked


def test_raster_shape_reads_the_header(dataset_config):
    pair = discover_pairs(dataset_config)[0]
    assert raster_shape(pair.image_path) == (96, 96)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_then_denormalize_round_trips():
    rgb = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    restored = denormalize_image(normalize_image(rgb, mean, std), mean, std)
    assert np.allclose(restored, rgb.astype(np.float32) / 255.0, atol=1e-6)


def test_normalize_moves_channels_first():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    assert normalize_image(rgb, (0, 0, 0), (1, 1, 1)).shape == (3, 4, 5)


def test_normalize_rejects_a_non_rgb_image():
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        normalize_image(np.zeros((4, 4), dtype=np.uint8), (0, 0, 0), (1, 1, 1))


# ---------------------------------------------------------------------------
# Step accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tiles,batch,accum,expected",
    [
        (0, 4, 2, 0),
        (4, 4, 1, 1),
        (10, 4, 2, 2),  # ceil(10/4)=3 micro, ceil(3/2)=2 steps
        (16, 8, 2, 1),
        (17, 8, 2, 2),  # the trailing partial group still takes a step
    ],
)
def test_expected_steps(tiles, batch, accum, expected):
    assert expected_steps(tiles, batch, accum) == expected


# ---------------------------------------------------------------------------
# The real DFC2019 Track-1 release: nested layout, no CRS, NaN-only invalids
# ---------------------------------------------------------------------------
#
# Verified on a real download: 1024x1024 tiles under Training-RGB/Track1-RGB and
# Training-Truth/Track1-Truth, CRS None, identity transform, no nodata tag, no
# sentinel value, rare non-finite AGL pixels and small finite negative heights.
# The rasters below are synthetic and only reproduce that HEADER shape.


def _no_crs_config(root, **overrides):
    kwargs = {
        "root": root,
        "tile_size": 32,
        "split": SplitConfig(mode="explicit", val_scene_ids=("x",)),
        **overrides,
    }
    return DatasetConfig(**kwargs)


def _no_crs_pair(root, stem="JAX_004_006", **pair_kwargs):
    write_synthetic_pair(root, stem, **{"size": 64, "georeferenced": False, **pair_kwargs})
    return discover_pairs(_no_crs_config(root))[0]


def test_default_layout_is_the_nested_real_one(tmp_path):
    cfg = _no_crs_config(tmp_path)
    assert cfg.image_dir == tmp_path / "Training-RGB" / "Track1-RGB"
    assert cfg.height_dir == tmp_path / "Training-Truth" / "Track1-Truth"


def test_discovers_pairs_in_the_nested_real_layout(tmp_path):
    root = tmp_path / "DFC2019"
    for stem in ("JAX_004_006", "JAX_004_007", "OMA_132_002"):
        write_synthetic_pair(root, stem, size=32, georeferenced=False)
    # Distractors present in the real download -- a CLS file beside the AGL
    # files, and a Validation folder with no truth -- must not disturb discovery.
    (root / DFC_HEIGHT_SUBDIR / "JAX_004_006_CLS.tif").write_bytes(b"")
    (root / "Validation" / "Track1").mkdir(parents=True)
    (root / "Validation" / "Track1" / "JAX_163_010_RGB.tif").write_bytes(b"")

    pairs = discover_pairs(_no_crs_config(root))
    assert [p.stem for p in pairs] == ["JAX_004_006", "JAX_004_007", "OMA_132_002"]
    assert [p.scene_id for p in pairs] == ["JAX_004", "JAX_004", "OMA_132"]
    assert [p.city for p in pairs] == ["JAX", "JAX", "OMA"]
    for pair in pairs:
        assert pair.image_path.parent == root / DFC_IMAGE_SUBDIR
        assert pair.height_path == root / DFC_HEIGHT_SUBDIR / f"{pair.stem}_AGL.tif"


def test_no_crs_pair_with_an_identical_grid_is_accepted(tmp_path):
    pair = _no_crs_pair(tmp_path / "ok")
    assert check_pair_registration(pair) is False  # accepted, not georeferenced
    sample = read_pair_tile(pair, CropWindow(0, 0, 32), _no_crs_config(tmp_path / "ok"))
    assert sample.valid.all()
    assert sample.metadata["georeferenced"] is False
    assert sample.metadata["crs"] == ""
    assert sample.metadata["transform"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_no_crs_pair_with_a_different_transform_is_refused(tmp_path):
    from rasterio.transform import Affine

    pair = _no_crs_pair(tmp_path / "shifted")
    _rewrite_height(pair, transform=Affine(1.0, 0.0, 2.0, 0.0, 1.0, 0.0))
    with pytest.raises(DatasetError, match="transforms differ"):
        check_pair_registration(pair)
    with pytest.raises(DatasetError, match="transforms differ"):
        read_pair_tile(pair, CropWindow(0, 0, 32), _no_crs_config(tmp_path / "shifted"))


@pytest.mark.parametrize("policy", ["error", "resample_height_to_image"])
def test_no_crs_pair_of_different_sizes_is_never_resampled(tmp_path, policy):
    """Without a footprint no resample can be verified, whatever the policy says."""
    root = tmp_path / policy
    pair = _no_crs_pair(root, height_size=32)
    cfg = _no_crs_config(root, on_size_mismatch=policy)
    with pytest.raises(DatasetError, match="sizes differ"):
        check_pair_registration(pair)
    with pytest.raises(DatasetError, match="sizes differ"):
        read_pair_tile(pair, CropWindow(0, 0, 32), cfg)


def test_reading_a_no_crs_pair_raises_no_not_georeferenced_warning(tmp_path):
    """rasterio's NotGeoreferencedWarning is silenced at the open, and only it."""
    import warnings

    from rasterio.errors import NotGeoreferencedWarning

    pair = _no_crs_pair(tmp_path / "quiet")
    cfg = _no_crs_config(tmp_path / "quiet")
    with warnings.catch_warnings(record=True) as caught:
        # Overrides the suite-wide ignore for this block, so a leak would show.
        warnings.simplefilter("always", NotGeoreferencedWarning)
        check_pair_registration(pair)
        raster_shape(pair.image_path)
        read_pair_tile(pair, CropWindow(0, 0, 32), cfg)
    assert not [w for w in caught if issubclass(w.category, NotGeoreferencedWarning)]


def test_open_raster_does_not_hide_other_warnings(tmp_path, monkeypatch):
    import warnings

    import depthwizard.relative.data as data_module

    pair = _no_crs_pair(tmp_path / "loud")
    real_open = data_module.rasterio.open

    def noisy_open(*args, **kwargs):
        warnings.warn("unrelated", UserWarning)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(data_module.rasterio, "open", noisy_open)
    with pytest.warns(UserWarning, match="unrelated"):
        with data_module.open_raster(pair.image_path):
            pass


@pytest.mark.parametrize("height_nodata", [-9999.0, None])
def test_nan_pixels_are_masked_without_a_nodata_tag(tmp_path, height_nodata):
    root = tmp_path / "nan"
    nan_at = [(3, 5), (20, 17), (31, 0)]
    pair = _no_crs_pair(root, fill_height=4.0, nan_pixels=nan_at)
    cfg = _no_crs_config(root, height_nodata=height_nodata)
    sample = read_pair_tile(pair, CropWindow(0, 0, 32), cfg)
    expected_invalid = np.zeros((32, 32), dtype=bool)
    for row, col in nan_at:
        expected_invalid[row, col] = True
    assert np.array_equal(~sample.valid, expected_invalid)
    assert np.isfinite(sample.height).all()  # NaN never reaches a tensor
    assert np.allclose(sample.height[sample.valid], 4.0)


@pytest.mark.parametrize("value", [-0.277, -3.1, -0.001])
def test_finite_negative_heights_stay_valid_and_unaltered(tmp_path, value):
    """Real AGL has small negative values; no lower bound, no clamp, no mask."""
    root = tmp_path / "neg"
    pair = _no_crs_pair(root, fill_height=value)
    sample = read_pair_tile(pair, CropWindow(0, 0, 32), _no_crs_config(root))
    assert sample.valid.all()
    assert np.allclose(sample.height, value)


def test_finite_negative_heights_give_a_finite_loss(tmp_path):
    torch = pytest.importorskip("torch")
    from depthwizard.relative.config import LossConfig
    from depthwizard.relative.loss import scale_invariant_loss

    root = tmp_path / "negloss"
    pair = _no_crs_pair(root, fill_height=-3.1, nan_pixels=[(0, 0)])
    sample = read_pair_tile(pair, CropWindow(0, 0, 32), _no_crs_config(root))
    target = torch.from_numpy(sample.height)[None, None]
    valid = torch.from_numpy(sample.valid)[None, None]
    out = scale_invariant_loss(torch.zeros_like(target), target, valid, LossConfig())
    assert torch.isfinite(out.loss)
    assert out.usable_count == 1


def test_heights_above_valid_max_are_still_masked_without_a_crs(tmp_path):
    root = tmp_path / "high"
    pair = _no_crs_pair(root, fill_height=5000.0)
    sample = read_pair_tile(pair, CropWindow(0, 0, 32), _no_crs_config(root))
    assert not sample.valid.any()


# ---------------------------------------------------------------------------
# Real tile geometry: 1024 px sources, 256 px crops
# ---------------------------------------------------------------------------


def test_1024_scene_gives_16_non_overlapping_validation_crops(tmp_path):
    windows = plan_validation_crops((1024, 1024), _no_crs_config(tmp_path, tile_size=256))
    assert len(windows) == 16
    assert {w.row_off for w in windows} == {0, 256, 512, 768}
    assert {w.col_off for w in windows} == {0, 256, 512, 768}
    covered = np.zeros((1024, 1024), dtype=int)
    for window in windows:
        covered[window.row_slice, window.col_slice] += 1
    assert covered.min() == 1 and covered.max() == 1  # exact partition, no padding


def test_training_crops_of_a_1024_scene_stay_within_0_to_768(tmp_path):
    cfg = _no_crs_config(tmp_path, tile_size=256)
    pair = ScenePair("JAX_004_006", "JAX_004", tmp_path / "a", tmp_path / "b", "JAX")
    rng = random.Random(0)
    windows = [plan_train_crops(pair, (1024, 1024), cfg, rng) for _ in range(2000)]
    rows = [w.row_off for w in windows]
    cols = [w.col_off for w in windows]
    assert min(rows) >= 0 and max(rows) <= 768
    assert min(cols) >= 0 and max(cols) <= 768
    assert all(w.size == 256 for w in windows)
    # The whole range is reachable, including both extremes.
    assert {0, 768} <= set(rows) | set(cols)


# ---------------------------------------------------------------------------
# per_city_scene: ~20% of tiles per city, views never cross the split
# ---------------------------------------------------------------------------


def _fake_pairs(tmp_path, tiles_per_city=None):
    """ScenePairs shaped like the real release (no files are needed to split).

    Tile ids and view counts vary per tile, and JAX_004 includes the real views
    JAX_004_006 and JAX_004_007.
    """
    tiles_per_city = tiles_per_city or {"JAX": 53, "OMA": 55}
    pairs = []
    for city, n_tiles in tiles_per_city.items():
        for t in range(n_tiles):
            scene = f"{city}_{4 + 3 * t:03d}"
            first_view = 4 + t % 3
            for view in range(first_view, first_view + 4 + t % 4):
                stem = f"{scene}_{view:03d}"
                pairs.append(ScenePair(stem, scene, tmp_path / f"{stem}_RGB.tif",
                                       tmp_path / f"{stem}_AGL.tif", city))
    return tuple(pairs)


def _sides(train, val):
    return {p.stem: "train" for p in train} | {p.stem: "val" for p in val}


@pytest.mark.parametrize("seed", range(25))
def test_views_of_one_tile_never_cross_the_split(tmp_path, seed):
    pairs = _fake_pairs(tmp_path)
    train, val = split_pairs(pairs, SplitConfig(mode="per_city_scene", seed=seed))
    assert len(train) + len(val) == len(pairs)
    assert not {p.scene_id for p in train} & {p.scene_id for p in val}
    sides = _sides(train, val)
    assert sides["JAX_004_006"] == sides["JAX_004_007"]


def test_per_city_split_holds_out_about_20_percent_of_each_city(tmp_path):
    pairs = _fake_pairs(tmp_path)
    train, val = split_pairs(pairs, SplitConfig(mode="per_city_scene", val_fraction=0.2))
    for city, n_tiles in (("JAX", 53), ("OMA", 55)):
        val_tiles = {p.scene_id for p in val if p.city == city}
        train_tiles = {p.scene_id for p in train if p.city == city}
        assert len(val_tiles) == round(0.2 * n_tiles) == 11
        assert len(train_tiles) == n_tiles - 11
        # Every view of a validation tile is in validation.
        assert {p.stem for p in pairs if p.scene_id in val_tiles} == {
            p.stem for p in val if p.city == city
        }


def test_per_city_split_is_deterministic_and_seed_controlled(tmp_path):
    pairs = _fake_pairs(tmp_path)
    split = SplitConfig(mode="per_city_scene", seed=20260918)
    first = _sides(*split_pairs(pairs, split))
    assert first == _sides(*split_pairs(pairs, split))
    # Independent of the order the pairs arrive in.
    assert first == _sides(*split_pairs(tuple(reversed(pairs)), split))
    other = _sides(*split_pairs(pairs, SplitConfig(mode="per_city_scene", seed=1)))
    assert other != first


def test_per_city_split_of_one_city_does_not_depend_on_the_others(tmp_path):
    both = _fake_pairs(tmp_path)
    jax_only = tuple(p for p in both if p.city == "JAX")
    split = SplitConfig(mode="per_city_scene")
    _, val_both = split_pairs(both, split)
    _, val_jax = split_pairs(jax_only, split)
    assert {p.scene_id for p in val_both if p.city == "JAX"} == {
        p.scene_id for p in val_jax
    }


def test_per_city_split_keeps_both_sides_for_every_city(tmp_path):
    pairs = _fake_pairs(tmp_path, {"JAX": 2, "OMA": 3})
    train, val = split_pairs(pairs, SplitConfig(mode="per_city_scene", val_fraction=0.9))
    for city in ("JAX", "OMA"):
        assert any(p.city == city for p in train)
        assert any(p.city == city for p in val)


def test_per_city_split_trains_on_a_single_tile_city_with_a_warning(tmp_path):
    from conftest import captured_depthwizard_logs

    pairs = _fake_pairs(tmp_path, {"JAX": 5, "OMA": 1})
    with captured_depthwizard_logs() as records:
        train, val = split_pairs(pairs, SplitConfig(mode="per_city_scene"))
    assert all(p.city == "JAX" for p in val)
    assert any(p.city == "OMA" for p in train)
    assert any("single scene id" in r.getMessage() for r in records)


def test_per_city_split_on_discovered_real_names(tmp_path):
    """End to end from filenames: JAX_004_006 and JAX_004_007 share one side."""
    root = tmp_path / "DFC2019"
    stems = [f"{city}_{tile:03d}_{view:03d}" for city in ("JAX", "OMA")
             for tile in (4, 17, 18, 20, 22) for view in (6, 7)]
    for stem in stems:
        write_synthetic_pair(root, stem, size=32, georeferenced=False)
    pairs = discover_pairs(_no_crs_config(root))
    for seed in range(10):
        train, val = split_pairs(pairs, SplitConfig(mode="per_city_scene", seed=seed))
        sides = _sides(train, val)
        assert sides["JAX_004_006"] == sides["JAX_004_007"]
        assert {p.city for p in train} == {p.city for p in val} == {"JAX", "OMA"}
