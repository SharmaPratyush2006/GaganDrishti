"""DFC2019 pair discovery and paired tile extraction.

This module is deliberately free of any PyTorch import. Everything that decides
*which* pixels are a training sample -- finding pairs on disk, splitting scenes,
choosing crop windows, reading the two rasters, building the validity mask --
lives here and is testable with numpy alone. :mod:`depthwizard.relative.dataset`
is the thin ``torch.utils.data.Dataset`` wrapper on top.

The one invariant
-----------------
**One crop window, applied to both rasters.** A sample's RGB pixels and its
height pixels are cut with the same row/column offsets, in image pixel space.
There is no code path that resizes the height raster on its own while leaving
the image alone. When the two rasters genuinely differ in size the
``dataset.on_size_mismatch`` policy decides between refusing (the default) and
one explicit, logged, nearest-neighbour resample of the *height* raster onto
the image grid -- recorded in the sample's metadata as ``height_resampled``, so
a downstream reader can see it happened.

Before any pixel is paired, the two rasters must describe the same ground (see
:func:`_check_registration`):

* **georeferenced pairs** need the same CRS, the same grid orientation and the
  same footprint. Equal pixel dimensions alone are not accepted as proof of
  alignment -- a same-sized pair offset by a few pixels would otherwise train
  silently on the wrong heights.
* **pairs with no CRS on either raster** -- which is what the real DFC2019
  Track-1 training release is: CRS None, identity transform, no GCPs/RPCs --
  have no footprint to compare, so the pixel grid is the only evidence of
  correspondence and it must match *exactly*: identical size and identical
  transform. Such a pair is never resampled, whatever ``on_size_mismatch``
  says, because there is no footprint against which a resample could be
  verified. The no-CRS condition is reported once per dataset, at
  construction (:class:`depthwizard.relative.dataset.RelativeHeightDataset`),
  not on every tile read.

Invalid pixels
--------------
A height pixel is invalid when it is non-finite, the raster's own nodata value,
the configured ``height_nodata`` value, above ``height_valid_max``, or inside
the zero-padding added to a scene smaller than one tile. The verified DFC2019
Track-1 AGL files have no nodata tag and no sentinel; their (rare) invalid
pixels are non-finite. There is no lower bound: a finite negative AGL value
such as -0.277 m is a real measurement and stays valid. Invalid pixels are
masked out of the loss. They are **never** imputed, interpolated or replaced
with a plausible number: DepthWizard does not fabricate target heights.

No DFC2019 data ships with this repository. :func:`discover_pairs` reads only
what the config points at, and raises :class:`DatasetError` naming the missing
directory when that path does not exist.
"""

from __future__ import annotations

import math
import random
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

from depthwizard.logging_setup import get_logger
from depthwizard.relative.config import DatasetConfig, SplitConfig

__all__ = [
    "DatasetError",
    "ScenePair",
    "CropWindow",
    "TileSample",
    "discover_pairs",
    "split_pairs",
    "raster_shape",
    "plan_train_crops",
    "plan_validation_crops",
    "read_pair_tile",
    "check_pair_registration",
    "open_raster",
    "sample_train_tile",
    "normalize_image",
    "denormalize_image",
    "summarize_pairs",
    "expected_steps",
]

log = get_logger(__name__)


class DatasetError(RuntimeError):
    """Raised when the configured dataset is missing, incomplete or unreadable."""


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenePair:
    """One RGB tile and its height tile, plus the scene they belong to."""

    #: Shared filename stem, e.g. ``JAX_004_007``.
    stem: str
    #: Group used for spatial splitting, e.g. ``JAX_004``: the geographic tile,
    #: shared by every view of it.
    scene_id: str
    image_path: Path
    height_path: Path
    #: City the tile belongs to, e.g. ``JAX``, from the regex's optional
    #: ``city`` group. Empty when the regex has no such group or did not match.
    city: str = ""


def _scene_id(stem: str, pattern: re.Pattern[str]) -> tuple[str, str]:
    """Recover ``(scene_id, city)`` from a filename stem.

    Falls back to the whole stem (and no city) when the pattern does not match,
    which makes every such file its own scene: safe (it can never merge two
    areas), and visible, because the caller is warned.
    """
    match = pattern.search(stem)
    if match is None:
        log.warning(
            "scene_id_regex did not match a filename stem; treating the stem as its "
            "own scene, which weakens the spatial split",
            extra={"stem": stem, "pattern": pattern.pattern},
        )
        return stem, ""
    city = match.group("city") if "city" in pattern.groupindex else None
    return match.group("scene"), city or ""


def discover_pairs(cfg: DatasetConfig) -> tuple[ScenePair, ...]:
    """Find every (RGB, height) pair under ``cfg.root``.

    Raises:
        DatasetError: if the root or either sub-directory is missing, if no
            image matches ``cfg.image_suffix``, or if any image has no
            corresponding height file. A half-present dataset is an error, not
            something to quietly train on.
    """
    if not cfg.root.exists():
        raise DatasetError(
            f"dataset.root does not exist: {cfg.root}\n"
            "DepthWizard ships no DFC2019 data. Download the IEEE GRSS DFC2019 "
            "Track 1 archive yourself and point dataset.root at it in "
            "configs/phase3.yaml."
        )
    image_dir, height_dir = cfg.image_dir, cfg.height_dir
    for label, directory in (("image_subdir", image_dir), ("height_subdir", height_dir)):
        if not directory.is_dir():
            raise DatasetError(
                f"dataset.{label} resolves to {directory}, which is not a directory. "
                f"Check dataset.root ({cfg.root}) and the sub-directory names against "
                "your actual DFC2019 layout."
            )

    pattern = re.compile(cfg.scene_id_regex)
    images = sorted(p for p in image_dir.iterdir() if p.name.endswith(cfg.image_suffix))
    if not images:
        raise DatasetError(
            f"no files ending in {cfg.image_suffix!r} found in {image_dir}. "
            "Adjust dataset.image_suffix to match your download."
        )

    pairs: list[ScenePair] = []
    missing: list[str] = []
    for image_path in images:
        stem = image_path.name[: -len(cfg.image_suffix)]
        height_path = height_dir / f"{stem}{cfg.height_suffix}"
        if not height_path.is_file():
            missing.append(f"{stem}{cfg.height_suffix}")
            continue
        scene_id, city = _scene_id(stem, pattern)
        pairs.append(
            ScenePair(
                stem=stem,
                scene_id=scene_id,
                image_path=image_path,
                height_path=height_path,
                city=city,
            )
        )

    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise DatasetError(
            f"{len(missing)} image(s) in {image_dir} have no matching height file in "
            f"{height_dir}: {shown}. Fix the layout or the suffixes rather than "
            "training on a partial dataset."
        )

    log.info(
        "discovered DFC2019-style pairs",
        extra={
            "root": str(cfg.root),
            "pairs": len(pairs),
            "scenes": len({p.scene_id for p in pairs}),
        },
    )
    return tuple(pairs)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_pairs(
    pairs: Sequence[ScenePair], split: SplitConfig
) -> tuple[tuple[ScenePair, ...], tuple[ScenePair, ...]]:
    """Divide pairs into ``(train, val)`` according to ``split``.

    Every mode except ``random_tile`` assigns whole scene ids, so all views of
    one geographic tile always land on the same side. ``per_city_scene``,
    ``scene_prefix`` and ``explicit`` are the spatially separated options; the
    two random modes are not, and :attr:`SplitConfig.is_spatially_separated`
    says so. ``random_tile`` returns the full pair list on both sides -- the
    tile-level shuffle happens in the dataset, and the caller is expected to
    have read the warning attached to it.
    """
    pairs = tuple(pairs)
    if not pairs:
        raise DatasetError("cannot split an empty pair list")

    if split.mode == "per_city_scene":
        val_scenes = _per_city_val_scenes(pairs, split)
        val = tuple(p for p in pairs if p.scene_id in val_scenes)
        train = tuple(p for p in pairs if p.scene_id not in val_scenes)
    elif split.mode == "scene_prefix":
        prefixes = tuple(split.val_scene_prefixes)
        val = tuple(p for p in pairs if p.scene_id.startswith(prefixes))
        train = tuple(p for p in pairs if not p.scene_id.startswith(prefixes))
    elif split.mode == "explicit":
        wanted = set(split.val_scene_ids)
        val = tuple(p for p in pairs if p.scene_id in wanted)
        train = tuple(p for p in pairs if p.scene_id not in wanted)
    elif split.mode == "random_scene":
        scenes = sorted({p.scene_id for p in pairs})
        rng = random.Random(split.seed)
        rng.shuffle(scenes)
        n_val = max(1, int(round(len(scenes) * split.val_fraction))) if split.val_fraction else 0
        val_scenes = set(scenes[:n_val])
        val = tuple(p for p in pairs if p.scene_id in val_scenes)
        train = tuple(p for p in pairs if p.scene_id not in val_scenes)
    else:  # random_tile
        log.warning(
            "split.mode='random_tile' is NOT a spatial split: validation tiles may "
            "neighbour training tiles and share the same buildings. Any score from "
            "it measures interpolation, not spatial generalisation.",
        )
        return pairs, pairs

    if not train:
        raise DatasetError(
            f"split.mode={split.mode!r} left the training set empty "
            f"({len(pairs)} pairs, {len({p.scene_id for p in pairs})} scenes). "
            "Check the held-out prefixes/ids against the scene ids actually present."
        )
    if not val:
        log.warning(
            "spatial split produced no validation scenes; training will run without "
            "validation",
            extra={"mode": split.mode, "scenes": len({p.scene_id for p in pairs})},
        )
    log.info(
        "split scenes",
        extra={
            "mode": split.mode,
            "spatially_separated": split.is_spatially_separated,
            "train_pairs": len(train),
            "val_pairs": len(val),
        },
    )
    return train, val


def _per_city_val_scenes(pairs: Sequence[ScenePair], split: SplitConfig) -> set[str]:
    """Pick ``round(val_fraction * n)`` scene ids per city for validation.

    Each city's scene ids are sorted, then shuffled by an RNG seeded from
    ``(split.seed, city)``: the choice is reproducible, independent of file
    order, and adding a city does not reshuffle the others. The count is
    clamped to ``[1, n - 1]`` so every city with at least two scenes feeds
    both sides. A city with a single scene cannot be split; it goes to
    training, with a warning.
    """
    scenes_by_city: dict[str, set[str]] = {}
    for pair in pairs:
        scenes_by_city.setdefault(pair.city, set()).add(pair.scene_id)

    val_scenes: set[str] = set()
    for city in sorted(scenes_by_city):
        scenes = sorted(scenes_by_city[city])
        if len(scenes) < 2:
            log.warning(
                "per_city_scene split: city has a single scene id and cannot "
                "contribute to validation; it is used for training only",
                extra={"city": city, "scene_id": scenes[0]},
            )
            continue
        n_val = min(len(scenes) - 1, max(1, round(len(scenes) * split.val_fraction)))
        # str seeds are hashed with SHA-512 by random.Random, not with the
        # per-process randomised str hash, so this is stable across runs.
        random.Random(f"{split.seed}:{city}").shuffle(scenes)
        val_scenes.update(scenes[:n_val])
        log.info(
            "per_city_scene split",
            extra={"city": city, "scenes": len(scenes), "val_scenes": n_val},
        )
    return val_scenes


# ---------------------------------------------------------------------------
# Crop geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CropWindow:
    """A crop in IMAGE pixel space, applied identically to both rasters."""

    row_off: int
    col_off: int
    size: int

    @property
    def row_slice(self) -> slice:
        return slice(self.row_off, self.row_off + self.size)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_off, self.col_off + self.size)


@contextmanager
def open_raster(path: str | Path) -> Iterator[Any]:
    """``rasterio.open(path)`` for reading, minus rasterio's ``NotGeoreferencedWarning``.

    The real DFC2019 Track-1 rasters have no CRS and no geotransform, so
    rasterio warns on every single open -- once per crop, per worker, per
    epoch. That condition is instead checked explicitly by
    :func:`_check_registration` and reported once per dataset. Only this one
    warning category, and only during the open (the only call that emits it),
    is silenced; every other warning passes through untouched.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        src = rasterio.open(path)
    try:
        yield src
    finally:
        src.close()


def raster_shape(path: Path) -> tuple[int, int]:
    """``(height, width)`` of a raster, read from its header only."""
    with open_raster(path) as src:
        return int(src.height), int(src.width)


def plan_train_crops(
    pair: ScenePair,
    shape: tuple[int, int],
    cfg: DatasetConfig,
    rng: random.Random,
) -> CropWindow:
    """Draw one random crop origin inside ``shape``.

    Origins are clamped so the window stays inside the raster; a scene smaller
    than ``tile_size`` yields offset 0 and is padded downstream (and only if
    ``allow_padding``). Rejection of mostly-invalid crops happens in
    :func:`read_pair_tile`'s caller, which can see the mask.
    """
    height, width = shape
    max_row = max(0, height - cfg.tile_size)
    max_col = max(0, width - cfg.tile_size)
    return CropWindow(
        row_off=rng.randint(0, max_row),
        col_off=rng.randint(0, max_col),
        size=cfg.tile_size,
    )


def plan_validation_crops(shape: tuple[int, int], cfg: DatasetConfig) -> tuple[CropWindow, ...]:
    """Every non-overlapping ``tile_size`` crop of a scene, in raster order.

    Deterministic by construction: no RNG, no shuffling, identical on every
    epoch and every machine, so two validation losses are comparable. The last
    row/column is pushed back from the edge rather than padded, matching
    :class:`depthwizard.ingest.tiling.Tiler`.
    """
    height, width = shape
    tile = cfg.tile_size

    def offsets(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        steps = list(range(0, extent - tile + 1, tile))
        if steps[-1] + tile < extent:
            steps.append(extent - tile)
        return steps

    return tuple(
        CropWindow(row_off=r, col_off=c, size=tile)
        for r in offsets(height)
        for c in offsets(width)
    )


# ---------------------------------------------------------------------------
# Paired reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileSample:
    """One spatially aligned (image, height, validity) triple plus its metadata."""

    #: ``(3, size, size)`` float32, normalised.
    image: np.ndarray
    #: ``(size, size)`` float32, in the ground truth's own units. Only the
    #: ground truth is ever in metres; the model's output is not.
    height: np.ndarray
    #: ``(size, size)`` bool. True where ``height`` is a real measurement.
    valid: np.ndarray
    #: Spatial bookkeeping: scene, crop origin, affine transform, CRS.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0


def normalize_image(rgb: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """``(H, W, 3)`` uint8/float -> ``(3, H, W)`` float32, scaled to [0,1] then standardised."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {array.shape}")
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)
    chw = np.ascontiguousarray(array.transpose(2, 0, 1))
    chw -= np.asarray(mean, dtype=np.float32)[:, None, None]
    chw /= np.asarray(std, dtype=np.float32)[:, None, None]
    return chw


def denormalize_image(chw: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`normalize_image`, for visualisation. Returns ``(H, W, 3)`` in [0,1]."""
    array = np.asarray(chw, dtype=np.float32).copy()
    array *= np.asarray(std, dtype=np.float32)[:, None, None]
    array += np.asarray(mean, dtype=np.float32)[:, None, None]
    return np.clip(array.transpose(1, 2, 0), 0.0, 1.0)


def _read_rgb_window(src: Any, window: Window) -> np.ndarray:
    """Read up to three bands of ``src`` over ``window`` as ``(h, w, 3)`` uint8-ish.

    A single-band source is replicated across RGB; a source with more than
    three bands contributes its first three. Both are logged the first time
    they matter, because either means the download is not the 3-band RGB
    product the defaults assume.
    """
    count = src.count
    if count >= 3:
        indexes = [1, 2, 3]
    elif count == 1:
        indexes = [1, 1, 1]
    else:
        raise DatasetError(
            f"{src.name}: expected a 1- or 3+-band image, got {count} bands"
        )
    data = src.read(indexes, window=window, boundless=True, fill_value=0)
    return np.ascontiguousarray(data.transpose(1, 2, 0))


#: Allowed disagreement between the two rasters' grids, as a fraction of a
#: pixel: same-size pairs must agree to 1/100 px (float round-off in the
#: GeoTIFF tags, nothing more); pairs of different size, which are resampled,
#: must cover the same ground to within half of the coarser pixel.
_SAME_GRID_TOLERANCE_PX = 0.01
_RESAMPLED_GRID_TOLERANCE_PX = 0.5


def _corner(transform: Any, col: float, row: float) -> tuple[float, float]:
    """``transform * (col, row)`` without the affine ``*`` operator.

    affine raises a PendingDeprecationWarning for ``*``, and this project runs
    its tests with warnings as errors.
    """
    return (
        transform.c + transform.a * col + transform.b * row,
        transform.f + transform.d * col + transform.e * row,
    )


@dataclass(frozen=True)
class _RasterGrid:
    """The header facts registration is checked on: CRS, transform, size."""

    crs: Any
    transform: Any
    height: int
    width: int

    @classmethod
    def of(cls, src: Any) -> "_RasterGrid":
        return cls(src.crs, src.transform, int(src.height), int(src.width))


def _check_registration(pair: ScenePair, image_src: _RasterGrid, height_src: _RasterGrid) -> bool:
    """Refuse an RGB/height pair that does not describe the same ground.

    Equal pixel dimensions are not enough: two rasters of the same size can
    sit in different CRSs, be offset by a few pixels, or have a different
    pixel size, and a model trained on such a pair learns heights for the
    wrong pixels without any error. So the pair must have

    * the same CRS, and never one without the other;
    * the same rotation/shear terms;
    * the same footprint: all four corners within
      :data:`_SAME_GRID_TOLERANCE_PX` of a pixel for same-size rasters, or
      :data:`_RESAMPLED_GRID_TOLERANCE_PX` of the coarser pixel for rasters
      that ``on_size_mismatch`` will resample onto each other.

    When **neither** raster has a CRS (the real DFC2019 Track-1 release) there
    is no footprint to compare, so the pixel grid must match exactly instead:
    identical ``(height, width)`` and an identical transform. No tolerance and
    no resampling: a no-CRS pair of different sizes is refused even under
    ``on_size_mismatch: resample_height_to_image``, since nothing could verify
    that the resampled grids cover the same ground.

    This function logs nothing; the caller reports the no-CRS condition once.

    Returns:
        True when the pair is georeferenced, False when neither raster has a
        CRS (and the exact-grid check passed).

    Raises:
        DatasetError: naming the pair and the disagreement.
    """
    image_crs, height_crs = image_src.crs, height_src.crs
    if bool(image_crs) != bool(height_crs):
        which = "image" if image_crs else "height"
        raise DatasetError(
            f"{pair.stem}: only the {which} raster has a CRS "
            f"(image={image_crs}, height={height_crs}). Cannot confirm the two "
            "rasters cover the same ground; refusing the pair."
        )
    if not image_crs:
        image_size = (image_src.height, image_src.width)
        height_size = (height_src.height, height_src.width)
        if image_size != height_size:
            raise DatasetError(
                f"{pair.stem}: neither raster has a CRS and their sizes differ "
                f"(image {image_size[1]}x{image_size[0]} px, height "
                f"{height_size[1]}x{height_size[0]} px). Without georeferencing the "
                "pixel grid is the only evidence the two correspond, so it must "
                "match exactly; such a pair is never resampled."
            )
        if tuple(image_src.transform)[:6] != tuple(height_src.transform)[:6]:
            raise DatasetError(
                f"{pair.stem}: neither raster has a CRS and their transforms differ "
                f"(image {tuple(image_src.transform)[:6]}, height "
                f"{tuple(height_src.transform)[:6]}). Without georeferencing the "
                "pixel grids must be identical; refusing the pair."
            )
        return False
    if image_crs != height_crs:
        raise DatasetError(
            f"{pair.stem}: image CRS {image_crs.to_string()} differs from height "
            f"CRS {height_crs.to_string()}. The pair is not spatially registered; "
            "reproject one onto the other's grid before training."
        )

    it, ht = image_src.transform, height_src.transform
    image_px = max(abs(it.a), abs(it.e), 1e-12)
    height_px = max(abs(ht.a), abs(ht.e), 1e-12)
    for name in ("b", "d"):
        if abs(getattr(it, name) - getattr(ht, name)) > 1e-9 * max(image_px, height_px):
            raise DatasetError(
                f"{pair.stem}: image and height transforms have different rotation "
                f"terms ({it} vs {ht}); the pair is not on a common grid."
            )

    same_size = (image_src.height, image_src.width) == (height_src.height, height_src.width)
    tolerance = (
        _SAME_GRID_TOLERANCE_PX * image_px
        if same_size
        else _RESAMPLED_GRID_TOLERANCE_PX * max(image_px, height_px)
    )
    image_corners = [
        _corner(it, c, r) for c, r in ((0, 0), (image_src.width, 0),
                                       (0, image_src.height),
                                       (image_src.width, image_src.height))
    ]
    height_corners = [
        _corner(ht, c, r) for c, r in ((0, 0), (height_src.width, 0),
                                       (0, height_src.height),
                                       (height_src.width, height_src.height))
    ]
    worst = max(
        max(abs(ix - hx), abs(iy - hy))
        for (ix, iy), (hx, hy) in zip(image_corners, height_corners)
    )
    if worst > tolerance:
        raise DatasetError(
            f"{pair.stem}: image and height rasters are misregistered -- their "
            f"footprints differ by up to {worst:.4g} CRS units (tolerance "
            f"{tolerance:.4g}; image pixel {image_px:.4g}). Image transform {tuple(it)[:6]}, "
            f"height transform {tuple(ht)[:6]}. Refusing to pair pixels that do "
            "not describe the same ground."
        )
    return True


def check_pair_registration(pair: ScenePair) -> bool:
    """Read both headers of ``pair`` and apply :func:`_check_registration`.

    Called once per scene when a dataset is built, so a misregistered pair
    fails at construction rather than midway through the first epoch.
    :func:`read_pair_tile` repeats the check on every read regardless.

    Returns:
        Whether the pair is georeferenced (see :func:`_check_registration`).
    """
    with open_raster(pair.image_path) as image_src, open_raster(
        pair.height_path
    ) as height_src:
        return _check_registration(
            pair, _RasterGrid.of(image_src), _RasterGrid.of(height_src)
        )


def read_pair_tile(
    pair: ScenePair,
    window: CropWindow,
    cfg: DatasetConfig,
) -> TileSample:
    """Cut ``window`` out of both rasters of ``pair``.

    The same ``(row_off, col_off, size)`` is used for the image and, after the
    documented size-mismatch handling, for the height raster. Regions outside
    the raster are zero-filled in the image and marked invalid in the mask.

    Raises:
        DatasetError: on a size mismatch under ``on_size_mismatch: error``, or
            on a scene smaller than one tile when ``allow_padding`` is false.
    """
    size = window.size
    with open_raster(pair.image_path) as image_src:
        image_shape = (int(image_src.height), int(image_src.width))
        if not cfg.allow_padding and (
            image_shape[0] < size or image_shape[1] < size
        ):
            raise DatasetError(
                f"{pair.stem}: scene is {image_shape[1]}x{image_shape[0]} px, smaller "
                f"than tile_size {size}, and dataset.allow_padding is false"
            )
        rio_window = Window(window.col_off, window.row_off, size, size)
        rgb = _read_rgb_window(image_src, rio_window)
        image_transform = image_src.window_transform(rio_window)
        image_crs = image_src.crs
        image_grid = _RasterGrid.of(image_src)

    with open_raster(pair.height_path) as height_src:
        # Before any pixel is paired: same CRS, same grid, same footprint --
        # or, with no CRS on either, an identical pixel grid.
        georeferenced = _check_registration(pair, image_grid, _RasterGrid.of(height_src))
        height_shape = (int(height_src.height), int(height_src.width))
        raster_nodata = height_src.nodata
        resampled = height_shape != image_shape
        if resampled:
            if cfg.on_size_mismatch == "error":
                raise DatasetError(
                    f"{pair.stem}: image is {image_shape[1]}x{image_shape[0]} px but "
                    f"height is {height_shape[1]}x{height_shape[0]} px. Refusing to "
                    "guess the correspondence. Either fix the data, or set "
                    "dataset.on_size_mismatch: resample_height_to_image to accept one "
                    "explicit nearest-neighbour resample of the height raster onto the "
                    "image grid."
                )
            # Explicit, documented: scale the IMAGE-space window into height
            # pixel space, then read it back at the image-space tile size with
            # nearest neighbour. Nearest because a height field has step
            # discontinuities at roof edges that interpolation would smear into
            # heights no surface ever had.
            row_scale = height_shape[0] / image_shape[0]
            col_scale = height_shape[1] / image_shape[1]
            height_window = Window(
                window.col_off * col_scale,
                window.row_off * row_scale,
                size * col_scale,
                size * row_scale,
            )
            log.warning(
                "height raster resampled onto the image grid (nearest neighbour)",
                extra={
                    "stem": pair.stem,
                    "image_shape": f"{image_shape[1]}x{image_shape[0]}",
                    "height_shape": f"{height_shape[1]}x{height_shape[0]}",
                },
            )
        else:
            height_window = Window(window.col_off, window.row_off, size, size)

        fill = float(cfg.height_nodata) if cfg.height_nodata is not None else float("nan")
        height = height_src.read(
            1,
            window=height_window,
            out_shape=(size, size),
            resampling=Resampling.nearest,
            boundless=True,
            fill_value=fill,
        ).astype(np.float32)

    valid = np.isfinite(height)
    if raster_nodata is not None:
        valid &= ~np.isclose(height, float(raster_nodata))
    if cfg.height_nodata is not None:
        valid &= ~np.isclose(height, cfg.height_nodata)
    if cfg.height_valid_max is not None:
        valid &= height <= cfg.height_valid_max

    # A pixel outside the source raster is padding, not measurement. Detect it
    # from geometry rather than from its value, so a real 0.0 m ground pixel at
    # the edge is not confused with a padded one.
    inside = np.zeros((size, size), dtype=bool)
    row_lo = max(0, -window.row_off)
    col_lo = max(0, -window.col_off)
    row_hi = min(size, image_shape[0] - window.row_off)
    col_hi = min(size, image_shape[1] - window.col_off)
    if row_hi > row_lo and col_hi > col_lo:
        inside[row_lo:row_hi, col_lo:col_hi] = True
    valid &= inside

    # Non-finite values must not survive into a tensor even where masked: a NaN
    # multiplied by a 0 mask is still NaN, and would poison the whole loss.
    height = np.where(valid, height, 0.0).astype(np.float32)

    metadata: dict[str, Any] = {
        "stem": pair.stem,
        "scene_id": pair.scene_id,
        "image_path": str(pair.image_path),
        "height_path": str(pair.height_path),
        "row_off": int(window.row_off),
        "col_off": int(window.col_off),
        "tile_size": int(size),
        "source_height": int(image_shape[0]),
        "source_width": int(image_shape[1]),
        "height_resampled": bool(resampled),
        "transform": [float(v) for v in tuple(image_transform)[:6]],
        "crs": image_crs.to_string() if image_crs else "",
        # False for the real DFC2019 Track-1 release: "transform" is then the
        # identity-based pixel grid, not a position on the ground.
        "georeferenced": bool(georeferenced),
    }
    return TileSample(
        image=normalize_image(rgb, cfg.normalize_mean, cfg.normalize_std),
        height=height,
        valid=valid,
        metadata=metadata,
    )


def sample_train_tile(
    pair: ScenePair,
    shape: tuple[int, int],
    cfg: DatasetConfig,
    rng: random.Random,
) -> TileSample:
    """Draw a training crop, retrying crops that are mostly nodata.

    Up to ``cfg.max_crop_attempts`` windows are tried; the first one meeting
    ``cfg.min_valid_fraction`` wins. If none does, the best attempt is returned
    **with its real mask** -- a poor crop still trains correctly because the
    loss only sees valid pixels, and dropping the sample entirely would bias
    the epoch towards dense areas.
    """
    best: TileSample | None = None
    for _ in range(cfg.max_crop_attempts):
        sample = read_pair_tile(pair, plan_train_crops(pair, shape, cfg, rng), cfg)
        if sample.valid_fraction >= cfg.min_valid_fraction:
            return sample
        if best is None or sample.valid_fraction > best.valid_fraction:
            best = sample
    assert best is not None  # max_crop_attempts >= 1 is enforced by the config
    return best


def summarize_pairs(pairs: Iterable[ScenePair]) -> dict[str, Any]:
    """Counts used in logs and in the training report."""
    pairs = tuple(pairs)
    scenes = sorted({p.scene_id for p in pairs})
    return {"pairs": len(pairs), "scenes": len(scenes), "scene_ids": scenes}


def expected_steps(n_tiles: int, batch_size: int, accumulation: int) -> int:
    """Optimiser steps in one epoch over ``n_tiles`` samples.

    Micro-batches are ``ceil(n_tiles / batch_size)``; an optimiser step happens
    every ``accumulation`` of them, plus one trailing step for a partial group
    (the trainer flushes leftover gradients at the end of the epoch rather than
    discarding them).
    """
    micro = math.ceil(n_tiles / batch_size) if n_tiles else 0
    return math.ceil(micro / accumulation) if micro else 0
