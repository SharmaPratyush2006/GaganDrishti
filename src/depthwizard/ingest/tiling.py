"""Tiling.

Large scenes are cut into fixed 512x512 tiles with a 64 px overlap. The
overlap exists because anything that looks at a neighbourhood -- and shadow
measurement certainly will -- produces garbage at a tile edge where the
neighbourhood is truncated. Overlapping tiles mean every output pixel is
interior to at least one tile.

Two pieces of bookkeeping make the tiles useful later:

**Geographic mapping.** Each tile carries its own affine transform and bounds,
derived from the parent raster's, so a tile can be georeferenced on its own and
a detection in tile coordinates converts back to map coordinates.

**Valid regions.** Overlap means pixels are covered more than once, so a
mosaic needs to know which tile *owns* each pixel. Each tile records a valid
span -- its share of the overlap, cut at the midpoint between neighbours --
and those spans **exactly partition** the source raster: no gaps, no
double-coverage. Stitching itself is a later phase; this module only records
what stitching will need, and :meth:`TileGrid.save_json` persists it.

Edge handling: tiles are kept at full ``tile_size`` by pushing the last tile
back from the edge rather than padding, so every tile is the same shape. The
cost is extra overlap in the final row/column, which the valid-span logic
absorbs correctly. A source smaller than one tile yields a single short tile.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from rasterio.transform import Affine

from depthwizard.ingest.geotiff import pixel_to_map, translate_transform
from depthwizard.logging_setup import get_logger

__all__ = [
    "Tile",
    "TileGrid",
    "Tiler",
    "DEFAULT_TILE_SIZE",
    "DEFAULT_OVERLAP",
]

log = get_logger(__name__)

DEFAULT_TILE_SIZE = 512
DEFAULT_OVERLAP = 64


@dataclass(frozen=True)
class Tile:
    """One tile: where it sits in the source, on the ground, and in a mosaic."""

    index: int
    tile_row: int
    tile_col: int

    #: Position and size within the source raster, in pixels.
    row_off: int
    col_off: int
    height: int
    width: int

    #: The tile's exclusive share of the source, in SOURCE pixel coordinates.
    #: Across a grid these spans exactly partition the raster.
    valid_row_start: int
    valid_row_end: int
    valid_col_start: int
    valid_col_end: int

    #: Six affine coefficients (a, b, c, d, e, f), or None if ungeoreferenced.
    transform: tuple[float, float, float, float, float, float] | None = None
    #: (west, south, east, north) in the source CRS, or None.
    bounds: tuple[float, float, float, float] | None = None

    # -- source-space slices -------------------------------------------------

    @property
    def row_slice(self) -> slice:
        return slice(self.row_off, self.row_off + self.height)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_off, self.col_off + self.width)

    @property
    def valid_row_slice(self) -> slice:
        return slice(self.valid_row_start, self.valid_row_end)

    @property
    def valid_col_slice(self) -> slice:
        return slice(self.valid_col_start, self.valid_col_end)

    # -- tile-space slices ---------------------------------------------------

    @property
    def local_valid_row_slice(self) -> slice:
        """The valid span expressed within this tile's own array."""
        return slice(self.valid_row_start - self.row_off, self.valid_row_end - self.row_off)

    @property
    def local_valid_col_slice(self) -> slice:
        return slice(self.valid_col_start - self.col_off, self.valid_col_end - self.col_off)

    @property
    def affine(self) -> Affine | None:
        """The tile's transform as an :class:`~rasterio.transform.Affine`."""
        return Affine(*self.transform) if self.transform else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TileGrid:
    """A complete tiling plan for one source raster."""

    tiles: tuple[Tile, ...]
    source_height: int
    source_width: int
    tile_size: int
    overlap: int
    n_tile_rows: int
    n_tile_cols: int
    crs: str | None = None
    source_transform: tuple[float, float, float, float, float, float] | None = None

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self) -> Iterator[Tile]:
        return iter(self.tiles)

    def __getitem__(self, index: int) -> Tile:
        return self.tiles[index]

    @property
    def stride(self) -> int:
        """Pixels between consecutive tile origins."""
        return self.tile_size - self.overlap

    def to_dict(self) -> dict[str, Any]:
        """Everything a future stitching step needs, as plain JSON types."""
        return {
            "source_height": self.source_height,
            "source_width": self.source_width,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "stride": self.stride,
            "n_tile_rows": self.n_tile_rows,
            "n_tile_cols": self.n_tile_cols,
            "crs": self.crs,
            "source_transform": list(self.source_transform) if self.source_transform else None,
            "tiles": [tile.to_dict() for tile in self.tiles],
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TileGrid":
        tiles = tuple(
            Tile(
                **{
                    **tile,
                    "transform": tuple(tile["transform"]) if tile.get("transform") else None,
                    "bounds": tuple(tile["bounds"]) if tile.get("bounds") else None,
                }
            )
            for tile in data["tiles"]
        )
        source_transform = data.get("source_transform")
        return cls(
            tiles=tiles,
            source_height=data["source_height"],
            source_width=data["source_width"],
            tile_size=data["tile_size"],
            overlap=data["overlap"],
            n_tile_rows=data["n_tile_rows"],
            n_tile_cols=data["n_tile_cols"],
            crs=data.get("crs"),
            source_transform=tuple(source_transform) if source_transform else None,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "TileGrid":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Axis geometry
# ---------------------------------------------------------------------------


def _axis_offsets(size: int, tile: int, stride: int) -> list[int]:
    """Tile origins along one axis, with the last one pushed back from the edge."""
    if size <= tile:
        return [0]
    offsets = list(range(0, size - tile + 1, stride))
    if offsets[-1] + tile < size:
        offsets.append(size - tile)
    return offsets


def _axis_valid_spans(offsets: Sequence[int], tile: int, size: int) -> list[tuple[int, int]]:
    """Exclusive ownership spans that exactly partition ``[0, size)``.

    Each boundary sits at the midpoint of the overlap between adjacent tiles,
    and each span starts where the previous one ended -- so the result is a
    partition even where the final tile overlaps its neighbour by more than
    ``overlap`` px.
    """
    spans: list[tuple[int, int]] = []
    previous_end = 0
    last = len(offsets) - 1
    for i, offset in enumerate(offsets):
        extent_end = min(offset + tile, size)
        end = size if i == last else (offsets[i + 1] + extent_end) // 2
        spans.append((previous_end, end))
        previous_end = end
    return spans


# ---------------------------------------------------------------------------
# Tiler
# ---------------------------------------------------------------------------


class Tiler:
    """Cuts rasters into overlapping fixed-size tiles.

    Args:
        tile_size: Tile edge length in pixels.
        overlap: Pixels shared between adjacent tiles. Must be < ``tile_size``.
    """

    def __init__(self, tile_size: int = DEFAULT_TILE_SIZE, overlap: int = DEFAULT_OVERLAP) -> None:
        tile_size = int(tile_size)
        overlap = int(overlap)
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= tile_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than tile_size ({tile_size}); "
                "otherwise tiles would not advance"
            )
        self.tile_size = tile_size
        self.overlap = overlap

    @property
    def stride(self) -> int:
        return self.tile_size - self.overlap

    def build_grid(
        self,
        height: int,
        width: int,
        *,
        transform: Affine | None = None,
        crs: str | None = None,
    ) -> TileGrid:
        """Plan a tiling of a ``height`` x ``width`` raster.

        No pixels are touched; this is pure geometry, so a grid can be planned
        before (or without ever) loading the imagery.
        """
        if height <= 0 or width <= 0:
            raise ValueError(f"raster size must be positive, got {height}x{width}")

        row_offsets = _axis_offsets(height, self.tile_size, self.stride)
        col_offsets = _axis_offsets(width, self.tile_size, self.stride)
        row_spans = _axis_valid_spans(row_offsets, self.tile_size, height)
        col_spans = _axis_valid_spans(col_offsets, self.tile_size, width)

        tiles: list[Tile] = []
        index = 0
        for tile_row, (row_off, (valid_row_start, valid_row_end)) in enumerate(
            zip(row_offsets, row_spans)
        ):
            tile_height = min(self.tile_size, height - row_off)
            for tile_col, (col_off, (valid_col_start, valid_col_end)) in enumerate(
                zip(col_offsets, col_spans)
            ):
                tile_width = min(self.tile_size, width - col_off)

                tile_transform = None
                bounds = None
                if transform is not None:
                    affine = translate_transform(transform, col_off, row_off)
                    tile_transform = tuple(affine)[:6]
                    west, north = pixel_to_map(affine, 0, 0)
                    east, south = pixel_to_map(affine, tile_width, tile_height)
                    bounds = (
                        min(west, east),
                        min(north, south),
                        max(west, east),
                        max(north, south),
                    )

                tiles.append(
                    Tile(
                        index=index,
                        tile_row=tile_row,
                        tile_col=tile_col,
                        row_off=row_off,
                        col_off=col_off,
                        height=tile_height,
                        width=tile_width,
                        valid_row_start=valid_row_start,
                        valid_row_end=valid_row_end,
                        valid_col_start=valid_col_start,
                        valid_col_end=valid_col_end,
                        transform=tile_transform,
                        bounds=bounds,
                    )
                )
                index += 1

        grid = TileGrid(
            tiles=tuple(tiles),
            source_height=height,
            source_width=width,
            tile_size=self.tile_size,
            overlap=self.overlap,
            n_tile_rows=len(row_offsets),
            n_tile_cols=len(col_offsets),
            crs=crs,
            source_transform=tuple(transform)[:6] if transform is not None else None,
        )
        log.info(
            "planned tile grid",
            extra={
                "source": f"{width}x{height}",
                "tiles": len(tiles),
                "layout": f"{grid.n_tile_rows}x{grid.n_tile_cols}",
                "tile_size": self.tile_size,
                "overlap": self.overlap,
            },
        )
        return grid

    def build_grid_for_scene(self, scene: Any) -> TileGrid:
        """Plan a tiling for a :class:`~depthwizard.ingest.loaders.LoadedScene`.

        Georeferencing is carried through when the scene has it, and omitted
        when it does not -- relative-mode inputs still tile fine, their tiles
        just have no map coordinates.
        """
        geo = scene.metadata.georeference
        return self.build_grid(
            height=scene.height,
            width=scene.width,
            transform=geo.transform if geo else None,
            crs=geo.crs.to_string() if geo else None,
        )

    @staticmethod
    def extract(array: np.ndarray, tile: Tile) -> np.ndarray:
        """Cut one tile's pixels out of a ``(rows, cols, ...)`` array."""
        return array[tile.row_slice, tile.col_slice, ...]

    def iter_tiles(
        self, array: np.ndarray, grid: TileGrid | None = None
    ) -> Iterator[tuple[Tile, np.ndarray]]:
        """Yield ``(tile, pixels)`` for every tile of ``array``."""
        if grid is None:
            grid = self.build_grid(array.shape[0], array.shape[1])
        if (grid.source_height, grid.source_width) != array.shape[:2]:
            raise ValueError(
                f"grid was planned for {grid.source_width}x{grid.source_height} "
                f"but array is {array.shape[1]}x{array.shape[0]}"
            )
        for tile in grid:
            yield tile, self.extract(array, tile)
