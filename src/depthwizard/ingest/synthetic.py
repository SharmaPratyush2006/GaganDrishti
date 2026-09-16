"""Synthetic fixture generator.

Produces a small, fully-known scene that DepthWizard can be developed and
tested against before any real satellite imagery is available:

* a flat ground plane,
* axis-aligned box buildings with **known** heights,
* shadows cast from a specified sun elevation and azimuth,
* written out as a georeferenced GeoTIFF with the illumination geometry stored
  in its metadata tags,
* plus a JSON sidecar and an optional height raster preserving the ground truth.

The point is that the answer is known in advance. Later phases can be scored
against this fixture without any hand labelling, and a bug in the geometry
shows up immediately as a mismatch against the sidecar.

Shadow rendering
----------------
A box's shadow is the union of its footprint swept from the base to the shadow
tip. The tip offset comes from :func:`depthwizard.physics.sun.shadow_pixel_offset`,
and the sweep is sampled at half-pixel steps so the swept region has no gaps.
This is deliberately the simplest model that is *geometrically correct* for a
flat scene: no self-shadowing between buildings, no penumbra, no atmosphere.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from depthwizard import __version__
from depthwizard.config import AppConfig, BuildingSpec, RasterConfig, SceneConfig, load_config
from depthwizard.ingest.geotiff import (
    MetadataTags,
    pixel_to_map,
    transform_from_raster_config,
    write_single_band,
)
from depthwizard.logging_setup import get_logger, setup_logging
from depthwizard.physics.sun import shadow_length_m, shadow_pixel_offset

__all__ = [
    "FootprintPixels",
    "SceneRender",
    "GeneratedFixture",
    "footprint_pixels",
    "render_scene",
    "generate_fixture",
    "main",
]

log = get_logger(__name__)

#: Sweep step, in pixels, used to rasterise a shadow. Below 1 px so the swept
#: footprint leaves no holes along diagonal shadow directions.
SHADOW_SWEEP_STEP_PX = 0.5


@dataclass(frozen=True)
class FootprintPixels:
    """A building footprint in pixel index space (half-open, like a slice)."""

    row_min: int
    row_max: int
    col_min: int
    col_max: int

    @property
    def n_rows(self) -> int:
        return self.row_max - self.row_min

    @property
    def n_cols(self) -> int:
        return self.col_max - self.col_min


@dataclass(frozen=True)
class SceneRender:
    """The rendered arrays for one scene, all sharing the raster's shape."""

    #: Reflectance-like float image in [0, 1].
    reflectance: np.ndarray
    #: Ground truth height in metres; 0 on bare ground.
    height_m: np.ndarray
    #: True where a building roof is visible.
    building_mask: np.ndarray
    #: True where ground is shadowed (excludes the buildings themselves).
    shadow_mask: np.ndarray
    #: Per-building footprints, keyed by building name.
    footprints: dict[str, FootprintPixels]


@dataclass(frozen=True)
class GeneratedFixture:
    """Paths and payload produced by :func:`generate_fixture`."""

    image_path: Path
    truth_path: Path
    height_path: Path | None
    scene: SceneConfig
    truth: dict[str, Any]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def footprint_pixels(building: BuildingSpec, raster: RasterConfig) -> FootprintPixels:
    """Convert a building's metre-based footprint to pixel indices.

    Raises:
        ValueError: if the footprint falls outside the raster, or rounds away
            to zero pixels at the configured GSD.
    """
    gsd = raster.gsd_m
    col_min = int(round(building.x_m / gsd))
    col_max = int(round((building.x_m + building.width_m) / gsd))
    row_min = int(round(building.y_m / gsd))
    row_max = int(round((building.y_m + building.depth_m) / gsd))

    if col_max <= col_min or row_max <= row_min:
        raise ValueError(
            f"building {building.name!r} is smaller than one pixel at gsd={gsd} m; "
            "increase its footprint or use a finer GSD"
        )
    if col_min < 0 or row_min < 0 or col_max > raster.width_px or row_max > raster.height_px:
        raise ValueError(
            f"building {building.name!r} footprint rows[{row_min}:{row_max}] "
            f"cols[{col_min}:{col_max}] falls outside the "
            f"{raster.height_px}x{raster.width_px} raster"
        )
    return FootprintPixels(row_min=row_min, row_max=row_max, col_min=col_min, col_max=col_max)


def _paint_rect(mask: np.ndarray, footprint: FootprintPixels, d_row: int, d_col: int) -> None:
    """Set ``mask`` True over ``footprint`` translated by (d_row, d_col), clipped."""
    n_rows, n_cols = mask.shape
    r0 = max(0, footprint.row_min + d_row)
    r1 = min(n_rows, footprint.row_max + d_row)
    c0 = max(0, footprint.col_min + d_col)
    c1 = min(n_cols, footprint.col_max + d_col)
    if r1 > r0 and c1 > c0:
        mask[r0:r1, c0:c1] = True


def _sweep_shadow(
    shadow_mask: np.ndarray,
    footprint: FootprintPixels,
    tip_offset: tuple[float, float],
) -> None:
    """Sweep ``footprint`` from its base to the shadow tip, marking the union."""
    d_row, d_col = tip_offset
    span_px = max(abs(d_row), abs(d_col))
    n_steps = max(1, int(np.ceil(span_px / SHADOW_SWEEP_STEP_PX)))
    for step in range(n_steps + 1):
        t = step / n_steps
        _paint_rect(shadow_mask, footprint, int(round(d_row * t)), int(round(d_col * t)))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_scene(scene: SceneConfig) -> SceneRender:
    """Render a scene's reflectance image, height truth and masks."""
    raster = scene.raster
    ground = scene.ground
    shape = (raster.height_px, raster.width_px)

    reflectance = np.full(shape, ground.reflectance, dtype=np.float32)
    height_m = np.zeros(shape, dtype=np.float32)
    building_mask = np.zeros(shape, dtype=bool)
    shadow_mask = np.zeros(shape, dtype=bool)
    footprints: dict[str, FootprintPixels] = {}

    for building in scene.buildings:
        footprint = footprint_pixels(building, raster)
        footprints[building.name] = footprint

        tip_offset = shadow_pixel_offset(
            height_m=building.height_m,
            sun_elevation_deg=scene.sun.elevation_deg,
            sun_azimuth_deg=scene.sun.azimuth_deg,
            gsd_m=raster.gsd_m,
        )
        _sweep_shadow(shadow_mask, footprint, tip_offset)
        _paint_rect(building_mask, footprint, 0, 0)

        rows = slice(footprint.row_min, footprint.row_max)
        cols = slice(footprint.col_min, footprint.col_max)
        height_m[rows, cols] = building.height_m

    # A building is not shadowed by itself, and its roof hides its own shadow.
    shadow_mask &= ~building_mask

    reflectance[shadow_mask] = ground.shadow_reflectance
    reflectance[building_mask] = ground.roof_reflectance

    if ground.noise_sigma > 0.0:
        rng = np.random.default_rng(ground.noise_seed)
        reflectance = reflectance + rng.normal(0.0, ground.noise_sigma, size=shape).astype(np.float32)

    np.clip(reflectance, 0.0, 1.0, out=reflectance)

    return SceneRender(
        reflectance=reflectance,
        height_m=height_m,
        building_mask=building_mask,
        shadow_mask=shadow_mask,
        footprints=footprints,
    )


# ---------------------------------------------------------------------------
# Ground truth sidecar
# ---------------------------------------------------------------------------


def _build_truth(scene: SceneConfig, render: SceneRender, transform) -> dict[str, Any]:
    """Assemble the ground truth record that accompanies the GeoTIFF.

    Everything here is either a configured input or an exact analytic
    consequence of one. Nothing is measured off the rendered image.
    """
    raster = scene.raster
    buildings: list[dict[str, Any]] = []

    for building in scene.buildings:
        footprint = render.footprints[building.name]
        length_m = shadow_length_m(building.height_m, scene.sun.elevation_deg)
        d_row, d_col = shadow_pixel_offset(
            height_m=building.height_m,
            sun_elevation_deg=scene.sun.elevation_deg,
            sun_azimuth_deg=scene.sun.azimuth_deg,
            gsd_m=raster.gsd_m,
        )
        centre_col = (footprint.col_min + footprint.col_max) / 2.0
        centre_row = (footprint.row_min + footprint.row_max) / 2.0
        easting, northing = pixel_to_map(transform, centre_col, centre_row)

        buildings.append(
            {
                "name": building.name,
                # The quantity the whole project exists to recover.
                "height_m": building.height_m,
                "footprint_m": {
                    "x_m": building.x_m,
                    "y_m": building.y_m,
                    "width_m": building.width_m,
                    "depth_m": building.depth_m,
                },
                "footprint_px": asdict(footprint),
                "centroid_easting_m": float(easting),
                "centroid_northing_m": float(northing),
                # Analytic, not measured: L = h / tan(elevation).
                "expected_shadow_length_m": length_m,
                "expected_shadow_length_px": length_m / raster.gsd_m,
                "shadow_tip_offset_px": {"d_row": d_row, "d_col": d_col},
            }
        )

    return {
        "depthwizard_version": __version__,
        "scene_name": scene.name,
        "synthetic": True,
        "raster": {
            "width_px": raster.width_px,
            "height_px": raster.height_px,
            "gsd_m": raster.gsd_m,
            "crs": raster.crs,
            "origin_easting_m": raster.origin_easting_m,
            "origin_northing_m": raster.origin_northing_m,
            "transform": list(transform)[:6],
        },
        "sun": {
            "elevation_deg": scene.sun.elevation_deg,
            "azimuth_deg": scene.sun.azimuth_deg,
            "azimuth_convention": MetadataTags.AZIMUTH_CONVENTION_VALUE,
        },
        "ground": asdict(scene.ground),
        "buildings": buildings,
    }


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------


def generate_fixture(
    scene: SceneConfig,
    out_dir: str | Path,
    *,
    run_id: str = "dev",
    write_height_raster: bool = True,
) -> GeneratedFixture:
    """Render ``scene`` and write the fixture to ``out_dir``.

    Writes:
        ``<name>.tif``              - uint8 single-band image (the "satellite" view)
        ``<name>_truth.json``       - ground truth sidecar
        ``<name>_truth_height.tif`` - float32 ground truth height raster (optional)

    Returns:
        A :class:`GeneratedFixture` describing what was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render = render_scene(scene)
    transform = transform_from_raster_config(scene.raster)
    truth = _build_truth(scene, render, transform)

    tags = {
        MetadataTags.SUN_ELEVATION: scene.sun.elevation_deg,
        MetadataTags.SUN_AZIMUTH: scene.sun.azimuth_deg,
        MetadataTags.AZIMUTH_CONVENTION: MetadataTags.AZIMUTH_CONVENTION_VALUE,
        MetadataTags.GSD: scene.raster.gsd_m,
        MetadataTags.SCENE_NAME: scene.name,
        MetadataTags.SYNTHETIC: "true",
        MetadataTags.PRODUCER: "depthwizard.ingest.synthetic",
        MetadataTags.VERSION: __version__,
        MetadataTags.RUN_ID: run_id,
    }

    # Round rather than truncate: a bare `astype` would bias every pixel down
    # by up to one digital number.
    image_u8 = np.clip(np.rint(render.reflectance * 255.0), 0, 255).astype(np.uint8)
    image_path = write_single_band(
        out_dir / f"{scene.name}.tif",
        image_u8,
        crs=scene.raster.crs,
        transform=transform,
        tags=tags,
    )

    height_path: Path | None = None
    if write_height_raster:
        height_path = write_single_band(
            out_dir / f"{scene.name}_truth_height.tif",
            render.height_m,
            crs=scene.raster.crs,
            transform=transform,
            tags={**tags, "CONTENT": "ground_truth_building_height_m"},
            nodata=None,
        )

    truth_path = out_dir / f"{scene.name}_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

    log.info(
        "generated synthetic fixture",
        extra={
            "scene": scene.name,
            "image": str(image_path),
            "truth": str(truth_path),
            "height_raster": str(height_path) if height_path else None,
            "buildings": len(scene.buildings),
            "gsd_m": scene.raster.gsd_m,
            "sun_elevation_deg": scene.sun.elevation_deg,
            "sun_azimuth_deg": scene.sun.azimuth_deg,
            "shadow_pixels": int(render.shadow_mask.sum()),
        },
    )

    return GeneratedFixture(
        image_path=image_path,
        truth_path=truth_path,
        height_path=height_path,
        scene=scene,
        truth=truth,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="depthwizard-make-fixture",
        description="Generate a synthetic shadow fixture GeoTIFF with known building heights.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"), help="YAML config file."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <paths.processed>/synthetic).",
    )
    parser.add_argument("--sun-elevation", type=float, default=None, help="Override sun elevation (deg).")
    parser.add_argument("--sun-azimuth", type=float, default=None, help="Override sun azimuth (deg).")
    parser.add_argument("--scene-name", type=str, default=None, help="Override the output scene name.")
    parser.add_argument(
        "--no-height-raster", action="store_true", help="Skip the ground truth height GeoTIFF."
    )
    parser.add_argument("--log-level", default=None, help="Override logging.level.")
    parser.add_argument("--log-format", default=None, choices=["text", "json"], help="Override logging.format.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``depthwizard-make-fixture``."""
    args = _parse_args(argv)

    config: AppConfig = load_config(args.config)
    log_config = config.logging
    if args.log_level or args.log_format:
        log_config = replace(
            log_config,
            level=args.log_level or log_config.level,
            format=args.log_format or log_config.format,
        )
    setup_logging(log_config)

    scene = config.scene
    if args.sun_elevation is not None or args.sun_azimuth is not None:
        scene = replace(
            scene,
            sun=replace(
                scene.sun,
                elevation_deg=(
                    args.sun_elevation if args.sun_elevation is not None else scene.sun.elevation_deg
                ),
                azimuth_deg=(
                    args.sun_azimuth if args.sun_azimuth is not None else scene.sun.azimuth_deg
                ),
            ),
        )
    if args.scene_name:
        scene = replace(scene, name=args.scene_name)

    out_dir = args.out_dir or (config.paths.processed / "synthetic")

    fixture = generate_fixture(
        scene,
        out_dir,
        run_id=config.project.run_id,
        write_height_raster=not args.no_height_raster,
    )

    print(f"image        : {fixture.image_path}")
    print(f"ground truth : {fixture.truth_path}")
    if fixture.height_path:
        print(f"height raster: {fixture.height_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
