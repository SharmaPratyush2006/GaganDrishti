#!/usr/bin/env python
"""Inspect an input: print its mode, CRS, GSD and sun geometry.

    python scripts/inspect_scene.py data/processed/synthetic/synthetic_city_a.tif

Add --require-absolute to fail loudly when the file cannot support metric
heights, instead of reporting a downgrade to relative mode. Add --tiles to also
print the tiling plan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from depthwizard.config import load_config  # noqa: E402
from depthwizard.ingest.metadata import MetadataError  # noqa: E402
from depthwizard.ingest.router import route  # noqa: E402
from depthwizard.ingest.tiling import Tiler  # noqa: E402
from depthwizard.logging_setup import setup_logging  # noqa: E402
from depthwizard.mode import Mode  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print ingest metadata for one input.")
    parser.add_argument("path", type=Path, help="GeoTIFF, PNG or JPG to inspect.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--require-absolute",
        action="store_true",
        help="Fail if the input cannot reach ABSOLUTE mode.",
    )
    parser.add_argument("--tiles", action="store_true", help="Also print the tiling plan.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config.logging)

    try:
        decision = route(args.path, require=Mode.ABSOLUTE if args.require_absolute else None)
    except MetadataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    metadata = decision.metadata
    print("=" * 72)
    print(metadata.summary())
    print("=" * 72)
    print(f"loader          : {decision.loader}")
    print(f"routing reason  : {decision.reason}")

    # The four values the Phase 1 acceptance check asks for, on their own.
    print("\n--- acceptance values ---")
    print(f"CRS           : {metadata.crs.to_string() if metadata.crs else '(none)'}")
    print(f"GSD           : {metadata.gsd_m if metadata.gsd_m is not None else '(none)'} m/px")
    print(f"sun elevation : {metadata.sun_elevation_deg if metadata.sun else '(none)'} deg")
    print(f"sun azimuth   : {metadata.sun_azimuth_deg if metadata.sun else '(none)'} deg")

    if args.tiles:
        tiler = Tiler(config.ingest.tiling.tile_size, config.ingest.tiling.overlap)
        geo = metadata.georeference
        grid = tiler.build_grid(
            metadata.height,
            metadata.width,
            transform=geo.transform if geo else None,
            crs=geo.crs.to_string() if geo else None,
        )
        print("\n--- tiling ---")
        print(
            f"{len(grid)} tile(s), layout {grid.n_tile_rows}x{grid.n_tile_cols}, "
            f"tile_size={grid.tile_size}, overlap={grid.overlap}, stride={grid.stride}"
        )
        for tile in list(grid)[:8]:
            print(
                f"  #{tile.index:<3} grid=({tile.tile_row},{tile.tile_col}) "
                f"origin=({tile.row_off},{tile.col_off}) size={tile.height}x{tile.width} "
                f"owns rows[{tile.valid_row_start}:{tile.valid_row_end}] "
                f"cols[{tile.valid_col_start}:{tile.valid_col_end}]"
            )
        if len(grid) > 8:
            print(f"  ... and {len(grid) - 8} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
