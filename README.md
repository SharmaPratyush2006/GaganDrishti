# DepthWizard

**Building height estimation from single-view satellite imagery, using shadow physics.**

Smart India Hackathon 2026 — problem statement **SIH26175**.

> **Status: Phases 0 and 1 (scaffolding + ingest) only.**
> There is no model, no shadow detector, no calibration, no DSM/DTM/nDSM and no
> viewer in this repository yet. See the [checklist](#implemented-vs-planned)
> for exactly what exists. **No accuracy metrics are reported anywhere in this
> repository, because none have been measured.**

---

## Objective

Recover the height of buildings from a **single** optical satellite image.

Height normally requires stereo pairs, lidar, or interferometric SAR — all of
which are expensive, infrequently captured, or unavailable for much of the
world. But a single image taken in daylight already contains a height signal
that is free: **the shadow each building casts**.

If you know how high the sun was and which way it was shining — and every
satellite product records this in its metadata — then the length of a shadow
determines the height of the object that cast it. DepthWizard's goal is to turn
that observation into a reliable, validated height product.

---

## Physics concept

For a vertical object of height `h` on flat ground, lit by a sun at elevation
`θ` above the horizon:

```
            *  <- sun at elevation θ
           /
          /
    +----+                     L = h / tan(θ)
    |    |  h                  h = L * tan(θ)
    |    | /|
    +----+--+-------
     <---- L ---->
      shadow on the ground
```

- **Forward model** (used to build the synthetic fixture): given a height,
  compute the shadow — `L = h / tan(θ)`.
- **Inverse model** (what later phases will use on real imagery): given a
  measured shadow, compute the height — `h = L · tan(θ)`.

The shadow's **direction** is fixed by the sun's azimuth: a shadow points
directly away from the sun, along bearing `azimuth + 180°`. A sun in the
south-east casts shadows to the north-west.

Both relationships live in `src/depthwizard/physics/sun.py` and are the only
physics implemented so far.

### Conventions (fixed project-wide, and recorded in every GeoTIFF we write)

| Quantity | Convention |
| --- | --- |
| `sun_elevation_deg` | Degrees above the horizon, in `(0, 90]`. 90° = overhead, no shadow. |
| `sun_azimuth_deg` | Compass bearing of the **sun**, degrees clockwise from North. 0 = N, 90 = E, 180 = S, 270 = W. |
| Rasters | North-up: column index increases East, row index increases South. |
| Units | A projected CRS in **metres**, so GSD and shadow lengths share units. |

### What this simple model ignores

Stated up front, because these are the error sources later phases must handle:
sloped terrain, shadows falling on other buildings rather than flat ground,
occlusion between neighbours, non-flat roofs, off-nadir viewing geometry,
penumbra softening, and shadows confused with dark roofs, water or asphalt.

---

## Architecture

```
                 ┌──────────────┐
  imagery ──────▶│    ingest    │  GeoTIFF I/O, synthetic fixtures
                 └──────┬───────┘
                        │  pixels + sun geometry
                 ┌──────▼───────┐
                 │   physics    │  L = h / tan(θ)  and its inverse
                 └──────┬───────┘
                        │  raw height estimates
                 ┌──────▼───────┐
                 │ calibration  │  terrain, view angle, per-scene bias
                 └──────┬───────┘
                        │  calibrated heights
                 ┌──────▼───────┐
                 │   surfaces   │  DSM / DTM / nDSM rasters
                 └──────┬───────┘
                        │  height products
                 ┌──────▼───────┐
                 │  validation  │  scoring against reference data
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │    viewer    │  web inspection UI
                 └──────────────┘
```

Cross-cutting, used by every stage: `config.py` (YAML + dataclasses),
`logging_setup.py` (structured logging) and `mode.py` (absolute vs relative).

Of the above, **only `ingest` and `physics` have any implementation.**
`calibration`, `surfaces`, `validation` and `viewer` are empty placeholders
that exist so import paths stay stable from the first commit.

### Absolute vs relative mode

Everything downstream branches on one question: can heights be expressed in
metres, or only relative to each other?

| | `Mode.ABSOLUTE` | `Mode.RELATIVE` |
| --- | --- | --- |
| Input | Georeferenced raster with sun angles | Plain PNG/JPG, or a raster missing metadata |
| Has | CRS, geotransform, GSD, sun elevation + azimuth | Pixels only |
| Yields | Heights in **metres** | Unitless, comparative heights |

A pixel shadow length only becomes a metre shadow length if you know the GSD,
and it only becomes a height if you know the sun elevation. Without both,
`h = L·tan(θ)` has no units to work with — hence the split.

---

## Repository layout

```
DepthWizard/
├── configs/
│   └── default.yaml              # the scene, sun, paths and logging config
├── data/
│   ├── raw/                      # source imagery          (git-ignored)
│   ├── processed/                # generated fixtures       (git-ignored)
│   └── outputs/                  # results                  (git-ignored)
├── scripts/
│   ├── make_synthetic_fixture.py # generate the synthetic fixture
│   └── inspect_scene.py          # print CRS / GSD / sun for any input
├── src/depthwizard/
│   ├── config.py                 # YAML -> validated dataclasses
│   ├── logging_setup.py          # text / JSON structured logging
│   ├── mode.py                   # Mode.ABSOLUTE / Mode.RELATIVE
│   ├── ingest/
│   │   ├── geotiff.py            # GeoTIFF read + write helpers
│   │   ├── synthetic.py          # synthetic fixture generator
│   │   ├── metadata.py           # provider-agnostic metadata discovery
│   │   ├── loaders.py            # GeoTIFFLoader, ImageLoader
│   │   ├── router.py             # picks the loader and the mode
│   │   ├── radiometry.py         # percentile stretch to 8-bit
│   │   └── tiling.py             # 512x512 tiles, 64 px overlap
│   ├── physics/
│   │   └── sun.py                # sun + shadow geometry
│   ├── calibration/              # NOT IMPLEMENTED
│   ├── surfaces/                 # NOT IMPLEMENTED
│   └── validation/               # NOT IMPLEMENTED
├── tests/
├── viewer/                       # NOT IMPLEMENTED
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Run the tests
pytest

# 3. Generate the synthetic fixture
python scripts/make_synthetic_fixture.py --config configs/default.yaml
```

Outputs land in `data/processed/synthetic/` (git-ignored):

| File | Contents |
| --- | --- |
| `synthetic_city_a.tif` | uint8 single-band GeoTIFF — the simulated "satellite" view |
| `synthetic_city_a_truth.json` | ground truth: heights, footprints, sun, grid |
| `synthetic_city_a_truth_height.tif` | float32 height raster, metres, 0 on bare ground |

Useful flags — sweep the sun without editing the config:

```bash
python scripts/make_synthetic_fixture.py --sun-elevation 25 --sun-azimuth 200 \
    --scene-name low_sun --log-format json
```

### Inspect any input

```bash
# Prints mode, CRS, geotransform, GSD and sun geometry (with provenance).
python scripts/inspect_scene.py data/processed/synthetic/synthetic_city_a.tif --tiles

# Refuse to proceed unless the file can support metric heights.
python scripts/inspect_scene.py some_scene.tif --require-absolute
```

```python
from depthwizard.ingest import load_scene, normalize_to_uint8, Tiler

scene = load_scene("data/processed/synthetic/synthetic_city_a.tif")
print(scene.mode)                      # Mode.ABSOLUTE
print(scene.metadata.crs)              # EPSG:32643
print(scene.metadata.gsd_m)            # 0.5
print(scene.metadata.sun_elevation_deg, scene.metadata.sun_azimuth_deg)  # 45.0 135.0

eight_bit = normalize_to_uint8(scene.array).array
grid = Tiler().build_grid_for_scene(scene)
```

---

## The synthetic fixture

Phase 0 deliberately starts with data whose answer is already known.

The generator builds a flat ground plane, places axis-aligned box buildings of
**specified** heights, computes each shadow from the configured sun elevation
and azimuth, and writes the result as a georeferenced GeoTIFF with the
illumination geometry stored in its metadata tags.

Because the heights were chosen rather than measured, later phases can be
scored against this fixture with **no hand labelling**, and a sign error or a
mixed-up azimuth convention shows up immediately as a mismatch.

Shadows are rendered by sweeping each footprint from its base to the shadow
tip, at half-pixel steps. This is the simplest model that is geometrically
correct for a flat scene — there is no self-shadowing between buildings, no
penumbra and no atmospheric scattering, and there is deliberately **no shadow
*detection*** anywhere in this repository.

---

## Ingest

### Metadata discovery

Providers store the same few numbers under different names in different places.
Sun elevation might be `SUN_ELEVATION` in a GDAL tag (Landsat), `meanSunEl` in
a Maxar `.IMD` sidecar, or implied by `MEAN_SUN_ZENITH_ANGLE` (Sentinel-2,
where elevation = 90 − zenith). Ingest probes all of them:

| Source | Examples |
| --- | --- |
| GDAL metadata domains | default, `IMD`, `RPC`, `IMAGERY`, `EXIF`, `TRE`, plus whatever the driver reports |
| RPC coefficients | rasterio's parsed RPCs and the `RPC` tag domain — recorded, not used for sun angles |
| `.IMD` sidecars | `scene.IMD` / `scene.tif.IMD` next to the image |

Key matching is case- and punctuation-insensitive, so one alias entry covers
`SUN_ELEVATION`, `sun-elevation`, `SunElevation` and `sunElevation`.

Two rules are enforced throughout:

**Nothing is invented.** Every resolved value carries a `Provenance` recording
the source, the key, the raw string, and any conversion applied:

```
sun elevation   : 45 deg  <- gdal:default[SUN_ELEVATION_DEG]=45.0
sun elevation   : 58 deg  <- gdal:default[MEAN_SUN_ZENITH_ANGLE]=32.0 (elevation_deg = 90 - zenith_deg)
```

**Missing metadata fails loudly.** A `MissingMetadataError` names every source
probed and every alias tried, so the message says what to fix:

```
missing required metadata: sun_elevation_deg, sun_azimuth_deg for no_sun.tif
  probed sources: gdal:default, gdal:IMAGE_STRUCTURE, gdal:DERIVED_SUBDATASETS
  keys tried for sun_elevation_deg: SUN_ELEVATION_DEG, SUN_ELEVATION, SUNEL, meanSunEl, ...
  hint: Supply the angles as GeoTIFF tags (SUN_ELEVATION_DEG / SUN_AZIMUTH_DEG),
        place the provider's .IMD sidecar next to the file, or load in relative
        mode with GeoTIFFLoader(require_absolute=False).
```

A georeferenced raster that is missing sun angles is *routed* to relative mode
rather than crashing — but never silently: the downgrade is logged at WARNING
and recorded on the metadata. Pass `require=Mode.ABSOLUTE` to make it fatal.

### Ground sample distance

GSD is always reported in **metres per pixel**, whatever the CRS says:

- **Projected CRS in metres** — taken straight from the transform.
- **Projected CRS in other units** (e.g. US survey feet) — converted via the
  CRS's linear-unit factor.
- **Geographic CRS** (degrees) — converted at the scene's centre latitude using
  the standard WGS84 series. This is an approximation, and the `gsd_note` field
  says so.
- **Rotated transforms** — pixel size is `hypot` over the transform's linear
  terms, so a rotated raster still reports its true pixel size.

A CRS with an identity transform is treated as *not* georeferenced, since
accepting it would silently imply a 1 m GSD.

### Radiometric normalisation

Imagery arrives as 8/11/12/16-bit. A **percentile stretch** (2–98 by default)
maps any of it to 8-bit. Percentiles rather than min/max because one saturated
pixel — glint off a roof — drags a min/max stretch to uselessness.

The per-band clip windows are returned, not just applied, because a stretch is
a lossy decision and you need the window to reproduce or invert it. Flat bands
are flagged `degenerate` and zeroed instead of dividing by zero; NaN and ±inf
are mapped deterministically rather than cast to garbage.

### Tiling

512×512 tiles with 64 px overlap. The overlap exists because anything that
looks at a neighbourhood — shadow measurement certainly will — produces garbage
where a tile edge truncates that neighbourhood.

Each tile carries its own affine transform and bounds, so a detection in tile
coordinates converts straight back to map coordinates.

Because overlap means pixels are covered more than once, each tile also records
a **valid span**: its exclusive share of the source, cut at the midpoint
between neighbours. These spans **exactly partition** the raster — no gaps, no
double-coverage — which is what a future stitching step needs. `TileGrid`
serialises the whole plan to JSON. *Stitching itself is a later phase; Phase 1
only records what it will need.*

Tiles stay at full `tile_size` by pushing the last one back from the edge
rather than padding, so every tile has the same shape. The final row/column
therefore overlaps by *more* than 64 px, never less, and the valid-span logic
absorbs that correctly.

## Configuration

`configs/default.yaml` maps one-to-one onto dataclasses in
`src/depthwizard/config.py`. Loading is **strict**: an unknown key or a missing
required key raises `ConfigError` rather than being silently ignored, so a typo
fails loudly instead of producing a subtly wrong run. Values are validated at
construction — a sun below the horizon or a shadow brighter than the ground is
rejected before any rendering happens.

## Logging

`logging_setup.setup_logging(config)` configures the `depthwizard` logger in one
of two shapes, chosen by `logging.format`:

- `text` — human-readable, structured fields appended as `key=value`
- `json` — one JSON object per line, for log tooling

Structured fields ride along on the standard `extra=` mechanism:

```python
log.info("generated synthetic fixture", extra={"scene": "city_a", "gsd_m": 0.5})
```

---

## Implemented vs planned

### Phase 0 — scaffolding ✅ IMPLEMENTED

- [x] Repository structure and packaging (`pyproject.toml`, `requirements.txt`)
- [x] YAML + dataclass configuration, with strict validation
- [x] Structured logging (text and JSON)
- [x] `.gitignore` excluding `data/raw`, `data/processed`, `data/outputs`
- [x] Sun/shadow geometry: `L = h / tan(θ)` and its inverse
- [x] GeoTIFF read/write helpers carrying sun metadata
- [x] Synthetic fixture generator (ground plane, box buildings, shadows,
      GeoTIFF, geospatial metadata, preserved ground truth heights)
- [x] Test suite covering all of the above

### Phase 1 — ingest & metadata ✅ IMPLEMENTED

- [x] `GeoTIFFLoader`: CRS, geotransform, GSD in m/px, sun elevation, sun azimuth
- [x] Metadata discovery across GDAL domains, RPC tags and `.IMD` sidecars
- [x] Provenance on every resolved value; loud failure on missing metadata
- [x] `ImageLoader` for PNG/JPG, with no geospatial assumptions
- [x] `Mode.ABSOLUTE` / `Mode.RELATIVE` and the mode router
- [x] Radiometric normalisation: percentile stretch to 8-bit
- [x] Tiling: 512×512, 64 px overlap, tile-to-geographic mapping, stitching metadata
- [x] Tests against the Phase 0 synthetic GeoTIFF

### Phase 2 — shadow extraction ❌ NOT IMPLEMENTED

- [ ] Shadow segmentation from imagery
- [ ] Building footprint association
- [ ] Shadow length measurement along the solar azimuth
- [ ] Tile stitching *(Phase 1 records the bookkeeping; nothing stitches yet)*

### Phase 3 — height estimation & calibration ❌ NOT IMPLEMENTED

- [ ] Height estimation from measured shadows
- [ ] Terrain slope and off-nadir view corrections
- [ ] Per-scene bias calibration
- [ ] Learned refinement model *(no model is built yet)*

### Phase 4 — surface products ❌ NOT IMPLEMENTED

- [ ] DSM generation
- [ ] DTM extraction
- [ ] nDSM (normalised height surface)
- [ ] External elevation sources (SRTM / CartoDEM)

### Phase 5 — validation ❌ NOT IMPLEMENTED

- [ ] Scoring against the synthetic fixture's ground truth
- [ ] Scoring against real reference DSM / lidar
- [ ] Error reporting *(no metrics are claimed until they are measured)*

### Phase 6 — viewer ❌ NOT IMPLEMENTED

- [ ] FastAPI service
- [ ] Web map viewer and per-building inspection (Three.js)

---

## Dependency note: GDAL

`pip install GDAL` compiles from source and needs the GDAL C++ SDK plus a
matching compiler; on Windows it fails out of the box. `rasterio` ships
prebuilt wheels that bundle their own GDAL, which is everything Phase 0 needs,
so GDAL is **not** a hard requirement here. If you need the standalone bindings
and CLI tools, install them via conda:

```bash
conda install -c conda-forge gdal
```

`torch` is declared as the `ml` extra rather than a core dependency, since
Phase 0 neither trains nor runs any model and the CUDA build is ~2.5 GB:

```bash
pip install -e ".[ml]"
```

---

## License

MIT
