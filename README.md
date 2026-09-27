# DepthWizard

**Building height estimation from single-view satellite imagery, using shadow physics.**

Smart India Hackathon 2026 — problem statement **SIH26175**.

> **Status: Phases 0–3 complete; Phase 4a (pixel-space metric calibration)
> implemented, but on real DFC2019 its calibration is largely unidentifiable.**
> Phase 3 was trained on real DFC2019 Track 1 and predicts **relative, unitless
> above-ground height (AGL)**. Phase 4a converts it to metres per tile,
> `AGL = a·exp(z_rel) + b`, anchored on building shadows, and scores it against
> DFC2019 AGL on held-out tiles. On real imagery, though, the measured shadow
> length barely tracks building height, so only 32 of 209 held-out tile fits
> have an identifiable scale. See the
> [Phase 4a section](#phase-4a--pixel-space-metric-calibration--implemented-real-data-calibration-largely-unidentifiable).
> Building footprints come from the DFC2019 **CLS labels** (class 6); nothing
> here *detects* buildings. There is **no georeferenced DSM, no COG export and
> no viewer** yet (Phase 4b / Phase 7).
>
> **The real-world figures in this repository are the Phase 4a DFC2019
> measurements, and each one carries its coverage.**
> - The primary one is a shadow-measured building evaluation: 1,762 evaluated
>   / 3,257 eligible / 20,558 total buildings, 99 % in Omaha.
> - No Jacksonville performance is claimed.
> - Figures against the synthetic fixture, whose heights were *specified*
>   rather than measured, are tagged `SYNTHETIC`.

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
                 │ calibration  │  relative AGL -> metres: AGL = a·exp(z_rel) + b
                 └──────┬───────┘
                        │  calibrated AGL (= nDSM)
                 ┌──────▼───────┐
                 │   surfaces   │  DSM = terrain + AGL, COG   (Phase 4b, planned)
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

Of the above, `ingest`, `physics`, `shadows`, `calibration` (Phase 4a, pixel
space) and `validation` have an implementation; `relative` (Phase 3) produces
the field `calibration` scales. `surfaces` and `viewer` are empty placeholders
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
│   ├── default.yaml              # the scene, sun, paths and logging config
│   ├── phase3.yaml               # relative-height training (template)
│   └── phase4.yaml               # Phase 4a calibration
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
│   ├── relative/                 # Phase 3: relative AGL model
│   ├── calibration/              # Phase 4a: pixel-space metric calibration
│   │   ├── footprints.py         # DFC2019 CLS class 6 -> footprints
│   │   ├── azimuth.py            # sun azimuth from shadow placement
│   │   ├── shadow_anchor.py      # references, sun scale, shadow heights
│   │   ├── fusion.py             # exp(z_rel), WLS a,b
│   │   ├── ransac.py             # seeded robust affine fit
│   │   ├── diagnostics.py        # error accounting, reports
│   │   ├── config.py             # configs/phase4.yaml loader
│   │   ├── ablation.py           # TRAINING-only threshold ablations, height census
│   │   └── run.py                # DFC2019 run + SYNTHETIC control
│   └── surfaces/                 # NOT IMPLEMENTED (Phase 4b)
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
> **nothing** about real-world accuracy (see Phase 4a for the DFC2019 figures).

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
- Losses and correlations above are not accuracy. Phase 3 on its own has no
  metric error; the first metric figures are Phase 4a's, after calibration.
- The split holds out tiles, not cities. Generalisation to an unseen city is
  untested.
- The unit tests still use small *synthetic* rasters and a toy CPU model; the
  real-data run above is a separate, manual verification.

### Phase 4a — pixel-space metric calibration 🟡 IMPLEMENTED; REAL-DATA CALIBRATION LARGELY UNIDENTIFIABLE

Code in `depthwizard.calibration`; config `configs/phase4.yaml`. Commands:

```bash
# TRAINING-only threshold ablations + per-city height census (no model needed)
python -m depthwizard.calibration.run ablation --config configs/phase4.yaml \
    --relative-config configs/phase3.local.yaml
# Frozen pipeline on training tiles (diagnostics only; Phase 3 was trained on them)
python -m depthwizard.calibration.run dfc --split-side train --views-per-scene 4 \
    --output-dir data/outputs/phase4a_train_diagnostics --config configs/phase4.yaml \
    --relative-config configs/phase3.local.yaml
# Final held-out evaluation (run once, after the method was frozen)
python -m depthwizard.calibration.run dfc --config configs/phase4.yaml \
    --relative-config configs/phase3.local.yaml
# SYNTHETIC control
python -m depthwizard.calibration.run synthetic --config configs/phase4.yaml
```

Outputs are git-ignored and live under `data/outputs/`:

| output | contents |
| --- | --- |
| `phase4a/phase4a_report.json` | report grouped by JAX / OMA / combined |
| `phase4a/per_tile.jsonl` | every tile, reference, fit and rejection, with additive error sums |
| `phase4a/figures/` | diagnostic figures |
| `phase4a_training_ablation/` | training ablation |
| `phase4a_train_diagnostics/` | training diagnostic run |
| `phase4a_v1_ed7efb8/` | the superseded v1 run, kept for provenance |

**Reading guide.** The results below come in four clearly separated kinds:
**TRAINING diagnostics** (where every threshold was chosen), the **HELD-OUT
evaluation** (run once, after the method was frozen), the **ORACLE** control
(diagnostic only), and the **SYNTHETIC** validation.

#### Summary

On real DFC2019, the shadow-length measurement **barely tracks building height**.
This is measured on training tiles:

- the median within-tile Spearman correlation of shadow length L with true
  height h is 0.10–0.20, whatever the gap tolerance, isolation policy or shadow
  threshold;
- 10–20 m buildings have shadows of the tile-median length (ratio ≈ 1.0), even
  though they are about 2× the tile-median height.

The diagnostic figures show why. The Phase 2 classical mask, even restricted to
CLS ground, marks most dark asphalt (streets, parking) as shadow, so rays
measure pavement extent. Also, in dense downtown JAX the tall buildings touch
the 512 px tile edge or cast shadows longer than the tile.

As a result, the per-tile scale `a` is **mostly not identifiable**: only 32 of
209 attempted held-out fits are significantly positive. The Phase 3 relative
field itself does carry height range. With true heights (the oracle), `a`
becomes significantly positive on tall-building tiles, so the dominant defect
is the shadow anchoring, not Phase 3. Low-rise geometry is a secondary limit.

**The previous v1 figure of 3.85 m dense MAE is not a valid height-model
result.** On most tiles `a` is statistically indistinguishable from zero, so
`a·exp(z_rel) + b` behaves close to a per-tile constant. The same caveat
applies to the v2 dense numbers below. They are reported because the protocol
requires them, not as a headline.

#### Method

Phase 3 predicts `z_rel ≈ log(AGL + 1) + c` (above-ground height, per-tile
offset `c`), so `AGL = a·exp(z_rel) + b`, with one `(a, b)` per 512 px tile.
Terrain is not a scale anchor. It is additive later, `DSM = T + AGL` (Phase 4b).

The sun is fixed **before** `a, b`, never jointly, because that system is
scale-degenerate. Per tile:

1. **Footprints**: CLS class 6 → 4-connected components → at least 50 px and
   not touching the tile edge. Class 6 ("building roof", ASPRS LAS) is verified
   against the official DFC2019 baseline code (`pubgeo/dfc2019`
   `track1-metrics.py`, `unets/params.py`), and the real CLS rasters contain
   exactly {2, 5, 6, 9, 17, 65}.
2. **Shadow mask**: the Phase 2 classical detector (image Otsu), unchanged,
   restricted to CLS ground (2).
3. **Sun azimuth**: footprint-adjacent directional shadow occupancy. It resolves
   the 180° ambiguity that PCA, gradient-histogram and Radon methods leave. The
   scan is over *sun* azimuths through the anti-sun function. Resolution bound
   is `atan(0.5/8)` = 3.58°.
4. **Shadow lengths**: Phase 2 `measure_shadow_lengths`, unchanged. **v2**:
   `gap_tolerance_px` = **2**, plus a new **merged-shadow** rule. If more than
   half the hit rays cross another building's footprint, the ray skipped over a
   neighbour and continued in its shadow, so the measurement is dropped. This is
   the same majority rule as edge truncation.
5. **Sun scale**: 2–3 automatically selected reference buildings.
   `metres_per_shadow_px = median(h_ref / L_px)` = `GSD·tanθ`.
   - Gates: NOMINAL; at least 200 px; solidity ≥ 0.8; ≥ 90 % valid AGL; at
     least 3 m; relative ray spread ≤ 0.5; shadow zone ≥ 80 % ground.
   - **v2**: the isolation gate is **off** (`isolation_px: 0`); contamination is
     handled by the merged-shadow rule.
   - Ranking is by relative spread, then area. It never uses AGL height, and a
     test checks this.
   - Each reference's ID and r̄ are recorded, and references are excluded from
     every fit and metric.
6. **Shadow heights** `hᵢ = k·Lᵢ,px`. FAILED constraints are excluded; REDUCED
   ones get weight × 0.5; NOMINAL get full weight. The weight is
   `wᵢ = 1/dhᵢ²` with `dh = k·ray_length_spread_px`. This is an **EMPIRICAL
   PROXY, not a measurement uncertainty**. No default `dh` is ever invented.
7. **Fit**: seeded RANSAC (inlier if |residual| ≤ 3·dhᵢ), then WLS. Every fit is
   **labelled**, never gated on `a > 0`:

   | label | meaning |
   | --- | --- |
   | `identifiable_positive` | a > 0 and a/SE(a) ≥ 2 (the conventional 95 % level, not tuned) |
   | `statistically_weak` | a > 0 but a/SE(a) < 2 |
   | `non_positive` | a ≤ 0 |
   | UNIDENTIFIABLE `insufficient_constraints` | fewer than 5 constraints; no `a, b` returned |
   | UNIDENTIFIABLE `numerically_unstable` | identical r̄ or condition > 10⁶; no `a, b` returned |

   SE(a) is the WLS standard error scaled by √(reduced χ²), because the proxy
   weights are only relative. Every fit reports the number of constraints, the
   weighted r̄ variance, the condition number, a, b, SE(a), SE(b), a/SE and the
   weighted residual.

**GSD** stays unverified (sources: ~30 cm, ~35 cm, 2048 px source tiles,
1.33 m), so `gsd_m: null`. Every height is GSD-invariant, and **tanθ and θ are
not determinable**. Even with a GSD, θ would only be an effective elevation.

#### Tuning decisions — TRAINING data only

All choices come from `run ablation`, which covers:
- 331 training images: 4 views per geographic tile, evenly spaced in sorted
  order;
- 86 geographic tiles (42 JAX, 44 OMA), giving 1,324 tiles of 512 px;
- quality criteria from training AGL:
  - within-tile Spearman of L vs h;
  - MAD of log(h/L);
  - leave-references-out error of `h = k·L`.

The held-out split was not used for any choice.

**v1 procedure error, corrected here:** v1 used a 4 px gap tolerance chosen
from held-out pass counts.

Comparison of variants (combined cities, image Otsu, merged-shadow check on
unless noted):

| variant | measured OK | tiles ≥ 2 refs | Spearman L~h | MAD log k | LRO MAE / MedAE (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| gap 2, isolation 10 | 54 % | 45 | 0.14 | 0.30 | 2.85 / 1.99 |
| **gap 2, isolation off (chosen)** | 54 % | **85** | 0.14 | 0.30 | 2.87 / 1.93 |
| gap 2, isolation off, merged check off | 54 % | 85 | 0.15 | 0.31 | 3.14 / 2.00 |
| gap 4, isolation off | 67 % | 95 | 0.16 | 0.32 | 2.80 / 1.91 |
| gap 6, isolation off | 73 % | 91 | 0.15 | 0.31 | 2.92 / 2.06 |
| ground-only Otsu, gap 2, isolation off | 54 % | 89 | 0.14 | 0.31 | 2.83 / 1.91 |

- **Gap 2 px.** 4 px vs 2 px was mixed: slightly higher Spearman and lower
  MAE, but higher k dispersion, all within noise. The pre-stated rule was to
  keep the validated Phase 2 default unless training showed a material gain.
  More pass-throughs are not better measurements: with the merged check *off*,
  larger gaps clearly degrade validity (Spearman 0.15 → 0.14 → 0.10 for
  2 / 4 / 6 px).
- **Isolation off.** Same validity, 1.9× the calibratable training tiles
  (JAX: 3 → 16). The isolation hypothesis for JAX was only a secondary cause:
  JAX's primary failure is the shadow measurement itself (next section).
  Directional isolation (only neighbours in the shadow path) rejected *more*
  than the 10 px rule and was dropped.
- **Merged-shadow check on.** It lowered leave-references-out MAE in every
  variant (3.14 → 2.87 m at gap 2).
- **k = median.** Inverse-variance had lower training MAE (2.63 vs 2.87 m) but
  is undefined when a reference has zero ray spread and depends on the proxy
  weights. The mean was worse than the median everywhere.
- **Shadow threshold unchanged.** Ground-only Otsu gave no validity gain.
- **Reference selection unchanged.** References only set k; their r̄ never
  enters the `a, b` fit. The diagnosis points to L–h validity, not reference
  choice.

#### Why Jacksonville fails, and other diagnostics

**Held-out tile funnel, by city:**

| stage | JAX | OMA |
| --- | ---: | ---: |
| 512 px tiles | 976 | 1,484 |
| tiles with kept footprints | 816 | 932 |
| sun azimuth recovered | 772 | 1,003 |
| reached reference selection | 667 | 849 |
| ≥ 2 valid references (sun calibrated) | **4** | 205 |
| a,b fit returned | 3 | 161 |
| fit `identifiable_positive` | **0** | 32 |

**JAX vs OMA, held-out:**

| diagnostic | JAX | OMA |
| --- | ---: | ---: |
| Phase 2 measurements succeeding | 35 % (2,576 / 7,342) | 63 % (6,734 / 10,718) |
| reference-candidate rejections: unusable shadow | 5,013 | 4,343 |
| reference-candidate rejections: not NOMINAL | 1,339 | 3,060 |
| reference-candidate rejections: ray spread | 234 | 2,016 |
| components below 50 px (speckle) | 73,479 of 87,053 | 14,136 of 29,484 |

- The JAX speckle holds only ~1 % of building pixels, so no buildings are lost
  to the min-area filter.
- JAX's primary blocker is the **shadow measurement**. Most rays find no
  continuous shadow before the gap limit.
- The JAX tall buildings are the downtown towers. They touch the tile edges,
  or their shadows are longer than the tile and fall on other buildings (see
  `figures/JAX_tallest__*.png`), so **none of the 28 held-out ≥ 20 m tiles can
  be calibrated**.

**Height census** (CLS components of at least 50 px, median AGL, one view per
geographic tile; description only):

| | buildings | median | mean | p75 | p90 | p95 | max | > 10 m | > 20 m | > 30 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train JAX | 2,167 | 4.2 | 5.2 | 6.0 | 8.1 | 9.8 | 127.0 | 106 | 25 | 13 |
| train OMA | 1,618 | 4.8 | 5.3 | 6.0 | 7.4 | 9.0 | 79.5 | 57 | 8 | 1 |
| held-out JAX | 584 | 4.0 | 4.9 | 5.6 | 7.4 | 9.5 | 53.0 | 28 | 8 | 4 |
| held-out OMA | 342 | 4.3 | 4.9 | 5.8 | 7.4 | 8.2 | 19.1 | 10 | 0 | 0 |

The buildings that actually became eligible references are all low-rise
(held-out: 891 eligible, median 4.35 m, max 8.2 m, none above 10 m). Only JAX
has the tall buildings, and none of them survive into calibration.

#### Tall-building identifiability test

Tile height group is the maximum footprint AGL. It is a diagnostic label only.

| group | set | shadow fit: ident. / attempted | shadow median a/SE | oracle fit: ident. / returned | oracle median a/SE |
| --- | --- | ---: | ---: | ---: | ---: |
| 0–5 m | held-out | 6 / 37 | 0.75 | 5 / 34 | 0.21 |
| 5–10 m | held-out | 20 / 136 | 0.70 | 37 / 96 | 1.47 |
| 10–20 m | held-out | 6 / 36 | 0.63 | **17 / 34** | **2.03** |
| ≥ 20 m | held-out | 0 calibrated (27 insufficient refs, 1 no azimuth) | — | — | — |
| 10–20 m | training* | 1 / 6 | −0.19 | 4 / 6 | 2.87 |

\*The training row is diagnostic: Phase 3 was trained on those tiles.

**Answer:**
- **Yes for the Phase 3 field.** With genuine height range, the oracle's `a`
  becomes meaningfully positive and identifiable: median a/SE rises from 0.21
  to 1.47 to 2.03 as the tile height group rises.
- **No for the shadow calibration.** Shadow-fit a/SE stays around 0.6–0.75 in
  every group.
- So the original problem is **primarily the shadow measurement (Phase 4)**,
  with low-rise geometry as a secondary factor.

#### HELD-OUT evaluation (method frozen before this run)

Split: the Phase 3 validation split, unchanged (`per_city_scene`, seed
20260918, tile-disjoint from training). It has 615 images and 22 geographic
tiles (11 JAX, 11 OMA), giving 2,460 tiles of 512 px (976 JAX, 1,484 OMA).

**Tile status by city:**

| status | JAX | OMA | combined |
| --- | ---: | ---: | ---: |
| no buildings | 160 | 552 | 712 |
| sun azimuth not recoverable | 149 | 83 | 232 |
| < 2 valid references | 663 | 644 | 1,307 |
| sun calibrated, fit UNIDENTIFIABLE (insufficient constraints) | 1 | 44 | 45 |
| fit returned | **3** (2 geographic tiles, 3 images) | **161** (6 geographic tiles, 117 images) | **164** |

**Identifiability of the fits returned** (combined):

| label | count |
| --- | ---: |
| `identifiable_positive` | **32** (JAX 0, OMA 32) |
| `statistically_weak` | 83 |
| `non_positive` | 49 |
| UNIDENTIFIABLE, insufficient constraints | 45 |
| UNIDENTIFIABLE, numerically unstable | 0 |

- Identifiable fraction: 32 / 209 attempted = **15 %** (32 / 164 returned =
  20 %).
- a: median 0.41 (p10–p90 −0.60 to 1.92).
- SE(a): median 0.59 (p10–p90 0.24 to 1.71).
- a/SE: median 0.68.
- For comparison, the oracle has 59 identifiable, 73 weak and 32 non-positive
  fits (median a/SE 1.43).

**References and buildings:**
- 506 calibration buildings (JAX 8, OMA 498).
- Evaluation buildings, shown as evaluated / eligible / total kept footprints:

| | evaluated | eligible | total kept footprints |
| --- | ---: | ---: | ---: |
| combined | 1,762 | 3,257 | 20,558 |
| JAX | 24 | 62 | 8,888 |
| OMA | 1,738 | 3,195 | 11,670 |

- Rejected evaluation buildings: 1,495 (1,388 FAILED, 66 edge-truncated, 37
  merged shadow, 4 with no proxy).

**Primary available metric — "shadow-measured building evaluation on
successfully sun-calibrated held-out tiles"** (shadow `hᵢ` vs median DFC2019
AGL; no `a, b` involved):

| | evaluated / eligible / total | MAE (m) | RMSE (m) | bias (m) | Spearman h_shadow~h |
| --- | ---: | ---: | ---: | ---: | ---: |
| combined | 1,762 / 3,257 / 20,558 | **2.63** | **4.65** | +0.53 | 0.25 |
| JAX | 24 / 62 / 8,888 | 4.79 | 10.16 | +2.94 | −0.13 |
| OMA | 1,738 / 3,195 / 11,670 | 2.60 | 4.53 | +0.50 | 0.26 |

By true height (combined, MAE / RMSE in m):

| true height | buildings | MAE / RMSE |
| --- | ---: | ---: |
| 0–5 m | 1,384 | 2.50 / 4.76 |
| 5–10 m | 363 | 3.04 / 4.24 |
| 10–20 m | 15 | 4.44 / 4.72 |
| ≥ 20 m | 0 | not yet measured |

Coverage limits:
- 99 % of these buildings are Omaha.
- They are 9 % of all kept footprints.
- Almost all are under 10 m tall.
- The JAX row rests on 24 buildings from 3 tiles, too few to describe JAX.

**Dense AGL, `a·exp(z_rel) + b` vs DFC2019 AGL** (reference pixels excluded).
This is *not* a valid height-model result; see the summary.

| | tiles | building MAE / RMSE | ground MAE / RMSE | all MAE / RMSE |
| --- | ---: | ---: | ---: | ---: |
| combined | 164 | 3.08 / 4.50 (4.59 M px) | 3.85 / 5.46 (37.8 M px) | 3.77 / 5.37 (42.4 M px) |
| JAX | 3 | 3.93 / 4.86 | 3.28 / 4.58 | 3.41 / 4.64 |
| OMA | 161 | 3.05 / 4.48 | 3.86 / 5.48 | 3.77 / 5.38 |
| combined, `identifiable_positive` fits only | 32 | 2.80 / 4.00 | 3.20 / 4.60 | 3.16 / 4.54 |
| combined, `statistically_weak` | 83 | 2.94 / 4.16 | 3.16 / 4.36 | 3.14 / 4.34 |
| combined, `non_positive` | 49 | 3.53 / 5.33 | 5.43 / 7.32 | 5.23 / 7.14 |

The identifiable subset is **not** clearly better than the weak one, so the
dense error is mostly per-tile offset error, not height structure.

Without RANSAC the same fits give 4.81 / 9.69 m (all pixels, combined). One
JAX tile's WLS fit is wild (JAX all-pixel 28.2 / 52.3 m), so RANSAC matters.

**ORACLE control — Phase 3 vs Phase 4 error decomposition** (diagnostic, NOT
deployable; the same 164 tiles, MAE / RMSE in m):

| pixels | Phase 3 + oracle a,b | Phase 3 + shadow a,b | calibration gap |
| --- | ---: | ---: | ---: |
| building | 1.72 / 2.71 | 3.08 / 4.50 | +1.36 / +1.79 |
| ground | 3.12 / 3.58 | 3.85 / 5.46 | +0.73 / +1.88 |
| all | 2.97 / 3.50 | 3.77 / 5.37 | +0.80 / +1.87 |

- On building pixels, the Phase 4 shadow calibration roughly doubles the error
  that remains with the oracle.
- On ground pixels, most of the error is already in Phase 3 under a per-tile
  affine map: the oracle itself has 3.12 m MAE, because a line through the
  buildings extrapolates poorly to the ground.

**Reference-count sensitivity** (combined, all pixels, MAE / RMSE in m):

| references | tiles fitted (without enough references) | own tiles | common 12 tiles* |
| --- | ---: | ---: | ---: |
| 2 | 164 (0) | 3.82 / 5.44 | 2.93 / 3.57 |
| 3 | 67 (97) | 3.21 / 4.48 | 3.00 / 3.74 |
| 5 | 12 (137; 15 fits failed) | 2.52 / 3.36 | 2.52 / 3.36 |

\*On the 12 tiles where all three counts were fitted, the top-5 union of
references is excluded for all counts. More references did not consistently
reduce error.

**Uncertainty forms:**
- Explicit `dh`: not measurable on DFC2019. None exists, so 209 of 209 fits are
  UNIDENTIFIABLE.
- Ray-spread proxy: all results above.

**v1 → v2 on the same held-out tiles:**
- Calibrated tiles went from 125 (JAX 0) to 164 (JAX 3).
- The dense figures are not comparable as a model result, for the
  identifiability reason above.

#### SYNTHETIC validation (specified, not measured, heights)

The Phase 0 fixture (sun 45° / 135°, GSD 0.5 m) is run with the **true** sun
elevation. The relative field is constructed as `z = log(AGL + 1) + 0.5`, so
the true values are a = 0.6065 and b = −1.

| check | result |
| --- | --- |
| azimuth | 135.0° recovered (error 0°) |
| azimuth sweep | 45 → 45°, 90 → 90°, 200 → 201°, 290 → 291° (bound 3.58°) |
| recovered a | 0.6124 |
| recovered b | −1.329 |
| fit label | `identifiable_positive` (a/SE = 113) |
| per-building errors | 0.30 / 0.02 / 0.25 / 0.34 m (the Phase 2 grid quantisation) |
| dense, building / ground | MAE 0.16 m / 0.32 m |
| ray-spread proxy | every spread is 0, so the fit is UNIDENTIFIABLE (insufficient constraints) |

The machinery is correct when shadows are clean and the metadata is known. The
real-data failure is a measurement-validity problem, not a solver bug.

#### Unresolved

- **Shadow measurement validity on DFC2019.** L does not track h. The classical
  mask includes pavement, and the unrectified views add layover. Fixing this
  needs a better shadow detector or geometry. That is new methodology and is
  not attempted here.
- **Tall buildings can't be used.** Tiles at 20 m and above are uncalibratable
  at 512 px (edge contact and long shadows). A larger calibration unit could
  help, but it conflicts with the per-tile Phase 3 offset.
- **Jacksonville is effectively unmeasured.** 3 tiles, 24 buildings, 0
  identifiable fits. No JAX performance is claimed.
- **tanθ / θ:** not determinable (GSD unverified).
- **DFC2019 sun-azimuth accuracy:** not measurable (no metadata).
- **Explicit dh:** unavailable for DFC2019.
- **Buildings ≥ 20 m on held-out tiles:** not yet measured.

### Phase 4b — georeferenced DSM ❌ NOT IMPLEMENTED (specification only)

Planned: DEM loading (SRTM / CartoDEM), CRS and overlap validation,
reprojection and resampling, additive terrain fusion `DSM = T + a·r + b`, COG
GeoTIFF writing with CRS and geotransform preserved, and verification on the
synthetic fixture. Programmatic COG checks come before any QGIS check. The
real georeferenced path is **NOT YET VERIFIED**. No DEM has been downloaded.

### Phase 5 — validation 🟡 INTERFACE IMPLEMENTED

- [x] Reference-data interface (`ReferenceSet`, JSON/CSV loading), requiring a
      stated source on every measurement
- [x] Scoring against the synthetic fixture's specified heights, tagged
      `SYNTHETIC` wherever it is reported
- [x] Error reporting and scatter plot, both absent where references are
- [ ] Manually measured real references through this interface *(none
      supplied; the real-world figures that exist are Phase 4a's, scored
      against DFC2019 lidar AGL)*

### Phase 6 — DTM extraction / nDSM ⛔ CANCELLED

Phase 3 predicts AGL, and **AGL is the nDSM**. There is no DTM extraction
(morphological, cloth, TIN) and no `DSM − DTM` step. The calibrated AGL field
is the above-ground product. The eventual DSM is `DEM + calibrated AGL`
(Phase 4b), and click-to-measure will read the calibrated AGL directly.

### Phase 7 — viewer ❌ NOT IMPLEMENTED

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
