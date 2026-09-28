"""Tiny GeoTIFF fixtures for the viewer's loader tests.

Written with the pipeline's own writers (``depthwizard.ingest.geotiff``) so the
JS loader is tested against exactly what the pipeline produces. These are NOT
terrain: they are small grids of known values with nodata at known pixels.

    .venv/Scripts/python.exe viewer/tests/fixtures/make_fixtures.py

``expected.json`` records every pixel as written (null = nodata), read back
with rasterio, so the JS tests compare against GDAL's view of the file.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine, from_origin

from depthwizard.ingest.geotiff import write_cog, write_single_band

HERE = Path(__file__).resolve().parent
CRS = "EPSG:32643"
TRANSFORM = from_origin(700000.0, 3170000.0, 0.5, 0.5)
NODATA = -9999.0


def _read_back(path: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            data = src.read(1, masked=True)
            mask = np.ma.getmaskarray(data)
            return {
                "width": src.width,
                "height": src.height,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "crs": src.crs.to_string() if src.crs else None,
                "transform": [float(v) for v in tuple(src.transform)[:6]],
                "tags": src.tags(),
                "values": [[None if mask[r, c] else float(data[r, c]) for c in range(src.width)]
                           for r in range(src.height)],
            }


def main() -> None:
    expected = {}

    # 1. Georeferenced float32 COG with nodata at known pixels and a VALID zero.
    h, w = 12, 20
    ndsm = (np.arange(h)[:, None] * 100 + np.arange(w)[None, :]).astype(np.float64) + 0.25
    ndsm[0, 0] = 0.0  # a real zero must stay valid
    for r, c in [(1, 1), (5, 7), (11, 19)]:
        ndsm[r, c] = np.nan  # -> nodata on disk
    path = HERE / "tiny_ndsm_cog.tif"
    write_cog(path, ndsm.astype(np.float32), crs=CRS, transform=TRANSFORM, nodata=NODATA,
              tags={"CONTENT": "nDSM = DSM - DTM: object height above the extracted ground",
                    "UNITS": "metres", "SYNTHETIC": "true", "PRODUCER": "viewer test fixture"})
    expected[path.name] = _read_back(path)

    # 2. No CRS, identity transform (pixel-space, like DFC2019 tiles): must be RELATIVE.
    plain = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    plain[2, 3] = NODATA
    path = HERE / "tiny_nocrs.tif"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        write_single_band(path, plain, crs=None, transform=Affine.identity(), nodata=NODATA,
                          tags={"CONTENT": "DSM: test grid without georeferencing", "UNITS": "metres"})
    expected[path.name] = _read_back(path)

    # 3. uint8 confidence-like raster with NO declared nodata (codes 0..4).
    conf = (np.arange(8 * 8).reshape(8, 8) % 5).astype(np.uint8)
    path = HERE / "tiny_confidence.tif"
    write_single_band(path, conf, crs=CRS, transform=TRANSFORM,
                      tags={"CONTENT": "Phase 5 confidence proxy (rule-based flags, NOT a probability)",
                            **{f"STATE_{i}": s for i, s in enumerate(["INVALID", "UNSUITABLE", "REDUCED", "NOT_ASSESSED", "HIGH"])}})
    expected[path.name] = _read_back(path)

    # 4. Sparse tiled file, no nodata: only the top-left 16x16 tile is written.
    path = HERE / "tiny_sparse.tif"
    profile = dict(driver="GTiff", width=32, height=32, count=1, dtype="uint8", crs=CRS, transform=TRANSFORM,
                   tiled=True, blockxsize=16, blockysize=16, sparse_ok=True, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((16, 16), 7, dtype=np.uint8), 1, window=((0, 16), (0, 16)))
    with rasterio.open(path) as src:
        counts = [int(src.get_tag_item(f"BLOCK_SIZE_{i}_{j}", "TIFF", bidx=1) or 0)
                  for j in range(2) for i in range(2)]
    expected[path.name] = {"width": 32, "height": 32, "written_block": [0, 16, 0, 16], "value": 7,
                           "block_byte_counts_row_major": counts}

    # 5-7. Source-image fixtures on a 4 x 6 grid (texture tests). Rows differ so a
    # vertical flip is detectable; the photometric tag is what the file declares.
    ih, iw = 4, 6
    gray = (np.arange(ih)[:, None] * 40 + np.arange(iw)[None, :]).astype(np.uint8)
    rgb = np.stack([np.arange(ih)[:, None] * 50 + np.zeros((1, iw)),
                    np.zeros((ih, 1)) + np.arange(iw)[None, :] * 40,
                    200 - np.arange(ih)[:, None] * 30 + np.zeros((1, iw))]).astype(np.uint8)
    images = {
        "tiny_gray_image.tif": (gray[None], "MINISBLACK"),
        "tiny_rgb.tif": (rgb, "RGB"),
        "tiny_multiband.tif": (rgb, "MINISBLACK"),
    }
    for name, (bands, photometric) in images.items():
        path = HERE / name
        with rasterio.open(path, "w", driver="GTiff", width=iw, height=ih, count=bands.shape[0], dtype="uint8",
                           crs=CRS, transform=TRANSFORM, photometric=photometric, compress="deflate") as dst:
            dst.write(bands)
        with rasterio.open(path) as src:
            expected[name] = {"width": iw, "height": ih, "bands": src.count, "photometric": photometric,
                              "colorinterp": [c.name for c in src.colorinterp],
                              "transform": [float(v) for v in tuple(src.transform)[:6]], "crs": src.crs.to_string(),
                              "pixels": src.read().transpose(1, 2, 0).tolist()}  # [row][col][band]

    (HERE / "expected.json").write_text(json.dumps(expected, indent=1), encoding="utf-8")
    print("wrote", ", ".join(expected))


if __name__ == "__main__":
    main()
