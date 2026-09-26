# DepthWizard

**Building height estimation from single-view satellite imagery, using shadow physics.**

Smart India Hackathon 2026 — problem statement **SIH26175**.

> **Status: Phases 0–2 complete; Phase 3 (relative-height baseline) trained on
> real DFC2019 data.** Shadow detection, shadow measurement and height
> estimation are implemented and tested. The Phase 3 model has been trained on
> the GPU and checked on held-out tiles, but its output is **relative and
> unitless — not metres** — and it has **no accuracy figure**. There is **no
> calibration, no DSM/DTM/nDSM and no viewer** yet, and **building footprints
> are supplied inputs — nothing in this
> repository detects buildings.** See the
> [checklist](#implemented-vs-planned) for exactly what exists.
>
> **No real-world accuracy has been measured, and none is claimed anywhere in
> this repository.** No reference measurements ship here. The only error figures
> that exist are against the synthetic fixture, whose heights were *specified*
> rather than measured, and every one of them is tagged `SYNTHETIC` in the code
> that produces it.

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
- **Inverse model** (used by Phase 2 on measured shadows): given a measured
  shadow, compute the height — `h = L · tan(θ)`.
- **Uncertainty**: `δh = tan(θ) · δL`. Exact rather than a linearisation, since
  `h` is linear in `L` once `θ` is fixed. It is only reported when the caller
  supplies a `δL`; no default error bar is invented.

The shadow's **direction** is fixed by the sun's azimuth: a shadow points
directly away from the sun, along bearing `azimuth + 180°`. A sun in the
south-east casts shadows to the north-west.

Geometry lives in `src/depthwizard/physics/sun.py`; the inverse estimator, the
uncertainty and the solar-elevation gate live in
`src/depthwizard/physics/height.py`.

### The 25–45° solar elevation band

`h = L · tan(θ)` is just as true at 15° as at 35°, but it is far worse
conditioned. Below ~25° shadows are long, and more likely to be occluded,
truncated by a tile edge, or to fall on ground that is not flat. Above ~45°
shadows are short and `tan(θ)` amplifies every pixel of measurement error into
more metres of height error.

DepthWizard therefore treats 25–45° as the **recommended** band, and the gate
is advisory: a scene outside it still gets a height, flagged
`Confidence.REDUCED` with the reason recorded alongside. **No scene is
rejected, and no height is silently corrected.**

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
                 │   shadows    │  detect mask, measure L along anti-sun
                 └──────┬───────┘
                        │  shadow lengths (+ SUPPLIED footprints)
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

The `shadows` package sits between `ingest` and `physics`: it turns pixels into
a shadow mask, measures each **supplied** footprint's shadow along the anti-sun
direction, and hands the length to `physics` for inversion.

Of the above, `ingest`, `physics`, `shadows` and `validation` have an
implementation. `calibration`, `surfaces` and `viewer` are empty placeholders
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
│   │   ├── sun.py                # sun + shadow geometry, anti-sun bearing
│   │   └── height.py             # h = L*tan(θ), δh, 25-45° band gate
│   ├── shadows/
│   │   ├── detector.py           # ShadowDetector ABC + classical HSV detector
│   │   ├── measure.py            # multi-ray shadow length along the anti-sun
│   │   └── pipeline.py           # detect -> measure -> invert
│   ├── validation/
│   │   ├── evaluation.py         # reference interface, metrics where they exist
│   │   └── plots.py              # predicted vs reference scatter
│   ├── calibration/              # NOT IMPLEMENTED
│   └── surfaces/                 # NOT IMPLEMENTED
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
penumbra and no atmospheric scattering.

### What the fixture can and cannot resolve

The sweep paints the footprint at whole-pixel translations, and the measurement
reads back the distance between two pixel centres, so a shadow length is only
recoverable to the resolution of the grid along its own direction.

That limit depends on the azimuth. Along a cardinal bearing the ray advances one
whole pixel per step and any whole-pixel length is exact. Along the 45° diagonal
consecutive pixel centres are √2 px apart **and every ray shares the same
phase**, so the recoverable lengths are exactly the multiples of √2 and no
amount of averaging across rays recovers a length between two of them.

The committed fixture is rendered at azimuth 135°, which is that worst case.
Its residual is therefore not an error in the estimator — see
[Phase 2 accuracy](#phase-2-accuracy-on-the-synthetic-fixture) for the
measured decomposition.

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

## Shadow physics

### The pipeline

```python
from depthwizard.shadows import (
    BuildingFootprint, DetectionContext, ShadowHeightPipeline,
)

context = DetectionContext.from_scene_metadata(scene.metadata)

# Footprints are SUPPLIED. Nothing here detects buildings.
footprints = [
    BuildingFootprint.from_bbox(
        "tower_a", shape=image.shape, row_min=80, row_max=120,
        col_min=80, col_max=120,
    )
]

result = ShadowHeightPipeline().run(image, footprints, context)
for estimate in result:
    print(estimate.building_id, estimate.height_m, estimate.confidence)
```

Three stages, each replaceable on its own:

1. **Detect** — `ClassicalShadowDetector` thresholds the HSV value channel,
   picking the cut with Otsu unless one is supplied. It reports the threshold it
   used and the fraction of the scene it marked, and flags a suspicious fraction
   rather than rejecting the scene.
2. **Measure** — every *shadow-facing* footprint pixel (the edges whose next
   step along the anti-sun direction leaves the footprint — picked out by
   geometry, not hard-coded) launches a ray. Rays tolerate small gaps, skip
   other buildings, and the per-building length is their **median**, so one ray
   down a dark alley cannot move the answer.
3. **Invert** — `h = L · tan(θ)`, with the band gate and any upstream
   complaints attached as confidence flags.

**Failure is a result, not a fallback.** If too few rays find shadow, the
measurement returns `ok=False` with a stated reason and `None` for the lengths.
Nothing substitutes a nominal value or a prior.

### Phase 2 accuracy on the synthetic fixture

> These are **synthetic** figures. The heights were specified when generating
> the fixture, not measured. They test this implementation's geometry and say
> **nothing** about real-world accuracy, which has not been measured.

Committed fixture — sun elevation 45°, azimuth 135°, GSD 0.5 m:

| building | L measured (px) | predicted (m) | specified (m) | abs. error (m) |
| --- | ---: | ---: | ---: | ---: |
| tower_a | 59.3970 | 29.698485 | 30.0 | 0.301 |
| block_b | 24.0416 | 12.020815 | 12.0 | 0.021 |
| slab_c  | 90.5097 | 45.254834 | 45.0 | 0.255 |
| low_d   | 11.3137 |  5.656854 |  6.0 | 0.343 |

MAE 0.230 m, max 0.343 m — against a grid bound of **0.354 m**
(`(√2/2) px · 0.5 m/px · tan 45°`).

That residual is the pixel grid, not the estimator. The same scene rendered at
azimuth 90°, where the geometry is exactly representable, recovers every height
to machine epsilon:

| building | L measured (px) | predicted (m) | specified (m) | abs. error (m) |
| --- | ---: | ---: | ---: | ---: |
| tower_a | 60.0000 | 30.000000 | 30.0 | 3.6 × 10⁻¹⁵ |
| block_b | 24.0000 | 12.000000 | 12.0 | 1.8 × 10⁻¹⁵ |
| slab_c  | 90.0000 | 45.000000 | 45.0 | 7.1 × 10⁻¹⁵ |
| low_d   | 12.0000 |  6.000000 |  6.0 | 8.9 × 10⁻¹⁶ |

Supporting evidence, all asserted in the test suite:

- the classical detector reproduces the fixture's shadow mask **exactly**, zero
  differing pixels, so segmentation contributes no error here;
- against analytic masks the measured length at azimuth 135° equals
  `round(L/√2)·√2` to 1e-14 — the quantization is predicted in closed form, not
  merely bounded;
- swept over many azimuths and lengths the mean error is +0.036 px, so the
  estimator is unbiased rather than systematically short.

## Validation

Validation is the interface through which **externally measured** heights
arrive, plus the reporting on top of it.

**No reference measurements ship with this repository, and none are generated.**
A `ReferenceMeasurement` cannot be constructed without a stated `source`, and
every `ReferenceSet` is tagged `SYNTHETIC` or `MANUAL_REAL` so a fixture number
can never be reported as a real-world one. A CSV carries no kind, so one must be
passed explicitly — that difference is exactly the thing that must never be
guessed.

With no references supplied, `EvaluationReport.metrics()` returns `None`, every
reference-derived field renders as `"not yet measured"`, and
`plot_predicted_vs_reference` writes no file and returns `None`. None of them
fall back to zero or to a default.

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

### Phase 2 — shadow physics ✅ IMPLEMENTED

- [x] `ShadowDetector` interface, so a learned segmenter can replace the
      classical one without changing a call site
- [x] `ClassicalShadowDetector`: HSV value-channel threshold, Otsu by default
- [x] Shadow length measurement along the anti-sun azimuth, by multi-ray
      casting with a robust (median) aggregator
- [x] Association of each measurement with its **supplied** footprint
- [x] Height estimation `h = L · tan(θ)` and uncertainty `δh = tan(θ) · δL`
- [x] 25–45° solar-elevation confidence gate (advisory, never a rejection)
- [x] `ShadowHeightPipeline` tying detect → measure → invert together
- [x] Reference-data interface and evaluation report, with metrics computed
      **only** where references exist
- [x] Predicted-vs-reference scatter plot, drawn only where references exist
- [x] Test suite covering all of the above

Not in Phase 2, and deliberately absent:

- **Building detection.** Footprints are supplied by the caller — drawn by
  hand, taken from a cadastral layer, or read from the fixture's ground truth.
- **Tile stitching.** Phase 1 records the bookkeeping; nothing stitches yet.

### Phase 3 — relative-height baseline 🟡 TRAINED ON REAL DATA, RELATIVE ONLY

Code lives in `depthwizard.relative`; config in `configs/phase3.yaml`.

- [x] DFC2019 pair discovery, 256×256 paired crops (one window cut from both
      rasters), nodata / padding masking, spatial (scene-level) validation split
- [x] RGB/AGL registration check: a pair must share a CRS, grid orientation
      and footprint (not just pixel dimensions), or it is refused
- [x] Valid *negative* AGL heights are kept as measurements; the loss's log
      transform is extended symmetrically below zero so they stay finite
- [x] Training crops change every epoch, including with the shipped
      `num_workers: 4` + `persistent_workers: true` (the epoch is shared with
      the worker processes)
- [x] Frozen encoder: DINOv2-Small (default) or ConvNeXt-Tiny (configured
      alternative), pretrained weights required unless random init is
      explicitly opted into. DINOv2's 14 px patches: the *network input* is
      resized 256 → 252; the target stays 256 and the prediction is upsampled
      back
- [x] DPT-style decoder (GroupNorm) and a single relative-height head (no
      semantic, shadow or uncertainty head)
- [x] Masked scale-invariant log loss, NaN-safe on invalid pixels
- [x] Training loop: bf16 when supported (explicit refusal or recorded fp32
      fallback otherwise), batch 8 × accumulation 2 = 16, per-epoch
      checkpoints carrying model / optimiser / scheduler state, config and
      measured losses
- [x] Per-tile inference (`python -m depthwizard.relative.inference`) with an
      RGB / ground truth / **Relative Height** figure; no metric conversion.
      Training tiles are 256 px; **inference tiles are 512 px**
      (`inference.tile_size`), run as a genuine 512 px forward pass (DINOv2
      input 504 = 36 patches), not an upsampled 256 px prediction
- [x] **Pretrained fallback** for when training stalls
      (`--fallback --fallback-reason "..."`): the raw output of a frozen,
      pretrained Depth Anything V2-Small (DINOv2-Small backbone). It is
      relative inverse depth, unitless, **not metres**, not trained on DFC2019,
      and labelled as such in every JSON sidecar and figure. Weights are never
      replaced with random ones
- [x] Random-init *architecture stub* (`build_fallback_model`) for offline
      tests only — labelled `random_init` everywhere, never "pretrained", and
      not the fallback above

**Environment (this machine):** Python 3.10 venv at `.venv` with torch
2.6.0+cu124 (CUDA available, RTX 4050 Laptop GPU, 6 GB, bf16 supported),
timm 1.0.30 and transformers 5.17.0. DepthWizard is installed editable
(`pip install -e .`), so `python -m depthwizard.relative.train` and
`python -m depthwizard.relative.inference` run without extra path setup. The
torch-dependent Phase 3 tests are run and pass on CPU. The pretrained
DINOv2-Small weights have been fetched once; the Depth Anything V2-Small
weights have not (its download test is opt-in: `DEPTHWIZARD_ALLOW_DOWNLOAD=1`).

**Training run (real data, this machine).** The shipped `configs/phase3.yaml`
settings, unchanged, on the real DFC2019 Track 1 training set (kept outside
Git): 2,783 RGB/AGL pairs, `per_city_scene` split into 2,168 training and 615
validation pairs (11 held-out tiles per city, tile-disjoint). 10 epochs × 1,084
optimiser steps on the RTX 4050 in bf16 (no substitution), 4 persistent
DataLoader workers, frozen pretrained DINOv2-Small
(`timm:vit_small_patch14_dinov2.lvd142m`, no fallback model), 548,609 trainable
and 21,654,912 frozen parameters. No OOM: peak allocated VRAM ≈ 297 MiB. About
34 minutes wall-clock.

| | epoch 0 | epoch 6 (best) | epoch 9 (last) |
|---|---|---|---|
| training loss | 0.432 | 0.267 | 0.256 |
| validation loss, 512 fixed crops | 0.421 | **0.335** | 0.340 |

Training loss fell every epoch. Validation loss fell overall, but it did not
fall every epoch and it levelled off after epoch 6. `best.pt` (epoch 6) and
every per-epoch checkpoint reload through `load_model_from_checkpoint`.

**Validation of `best.pt` on the full held-out grid.** All 9,840 validation
crops scored, none skipped: **validation loss 0.317**. This is the training
loss function (scale-invariant log loss, λ = 0.5). It is **not an accuracy**.

**Descriptive sanity check on 8 held-out tiles** (4 JAX, 4 OMA, 512 px):
per-tile Pearson correlation between the prediction and the loss's
log-height of the ground truth ranges from 0.55 to 0.91 on the 7 tiles that
have height structure. Spearman correlation ranges from 0.59 to 0.79. These
are correlations, **not accuracy percentages**, and they say nothing about
metric DSM accuracy. The eighth tile (an airport apron with ground truth
within ±0.005 m) has no height structure, so its correlation (≈ 0) carries no
information. In the figures, buildings, trees and elevated roads appear where
the ground truth has them. Edges are blurred and the relative height of very
tall structures is compressed.

**Still open:**

- The output is **relative and unitless**. It is not metres, and each tile has
  its own unknown offset, so tiles are not stitched. Converting it to metres
  is Phase 4.
- **No accuracy exists.** Losses and correlations above are not accuracy, and
  no metric error has been measured.
- The split holds out tiles, not cities. Generalisation to an unseen city is
  untested.
- The unit tests still use small *synthetic* rasters and a toy CPU model; the
  real-data run above is a separate, manual verification.

### Phase 4 — calibration & surface products ❌ NOT IMPLEMENTED

- [ ] Relative → metric calibration (shadow geometry, SRTM / CartoDEM)
- [ ] Terrain slope and off-nadir view corrections
- [ ] Per-scene bias calibration
- [ ] DSM generation
- [ ] DTM extraction
- [ ] nDSM (normalised height surface)
- [ ] External elevation sources (SRTM / CartoDEM)

### Phase 5 — validation 🟡 INTERFACE IMPLEMENTED, NO REAL DATA

- [x] Reference-data interface (`ReferenceSet`, JSON/CSV loading), requiring a
      stated source on every measurement
- [x] Scoring against the synthetic fixture's specified heights, tagged
      `SYNTHETIC` wherever it is reported
- [x] Error reporting and scatter plot, both absent where references are
- [ ] Scoring against real reference DSM / lidar *(**no real reference data has
      been supplied to this repository**, so no real-world accuracy exists)*

### Phase 6 — viewer ❌ NOT IMPLEMENTED

- [ ] FastAPI service
- [ ] Web map viewer and per-building inspection (Three.js)

---

## Dependency note: GDAL

`pip install GDAL` compiles from source and needs the GDAL C++ SDK plus a
matching compiler; on Windows it fails out of the box. `rasterio` ships
prebuilt wheels that bundle their own GDAL, which is everything implemented
so far needs,
so GDAL is **not** a hard requirement here. If you need the standalone bindings
and CLI tools, install them via conda:

```bash
conda install -c conda-forge gdal
```

`torch`, `timm` and `transformers` (for the Phase 3 pretrained fallback) are
declared as the `ml` extra rather than core dependencies. Only Phase 3 needs
them, and the CUDA wheel is a 2.53 GB download before it is unpacked. Install
the CUDA build first, so pip does not pull a CPU wheel:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[ml]"
```

---

## License

MIT
