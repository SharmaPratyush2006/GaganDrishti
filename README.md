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
> here *detects* buildings. **Phase 4b** (georeferenced DSM = DEM + calibrated
> AGL, exported as COG) is implemented and validated on the **synthetic**
> fixture only; the real-data georeferenced path is **NOT YET VERIFIED**.
> **Phase 5** (validation harness) is implemented. It has been run as a
> **city-held-out model evaluation with LiDAR-anchored per-tile metric
> calibration**: the model was trained on Jacksonville (JAX) only and evaluated
> on Omaha (OMA). This is **not** an unseen-city zero-shot result, because OMA
> lidar sets each tile's metric scale. Only 714 of 7,072 OMA tiles could be
> calibrated (see [Phase 5](#phase-5--validation-harness--implemented)). There
> is no viewer yet (Phase 7).
>
> **The real-world figures in this repository are DFC2019 measurements, and
> each one carries its coverage.**
> - Phase 4a (model trained on both cities, tile-held-out): a shadow-measured
>   building evaluation, 1,762 evaluated / 3,257 eligible / 20,558 total
>   buildings, 99 % in Omaha.
> - Phase 5 (model trained on JAX only, OMA held out): dense AGL on the 714
>   calibrated OMA tiles (10 % of OMA tiles). The pooled numbers are dominated
>   by per-tile calibration error, not model quality; see Phase 5.
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
                 │   surfaces   │  DSM = terrain + AGL, COG   (Phase 4b, synthetic only)
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
│   ├── phase4.yaml               # Phase 4a calibration
│   └── phase5.yaml               # Phase 5 validation harness
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
│   │   ├── plots.py              # predicted vs reference scatter
│   │   ├── config.py             # Phase 5: configs/phase5.yaml loader
│   │   ├── masks.py              # Phase 5: validity mask, terrain/building masks
│   │   ├── metrics.py            # Phase 5: MAE, RMSE, Pearson r, δ thresholds
│   │   ├── spatial_split.py      # Phase 5: geographic split audit vs checkpoint
│   │   ├── confidence.py         # Phase 5: shadow / water / sun-band proxy
│   │   ├── error_map.py          # Phase 5: error GeoTIFF + PNG
│   │   ├── report.py             # Phase 5: Markdown report
│   │   └── run.py                # Phase 5: one-command harness
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
│   │   ├── run.py                # DFC2019 run + SYNTHETIC control
│   │   └── terrain.py            # Phase 4b: DEM -> image grid (reproject, validate)
│   └── surfaces/                 # Phase 4b: DSM = T + a*exp(z_rel) + b
│       ├── dsm.py                # AGL, DSM fusion, nodata propagation
│       └── run.py                # SYNTHETIC DSM export + COG validation
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

### Phase 4b — georeferenced DSM export ✅ IMPLEMENTED, SYNTHETIC ONLY

> **Phase 4b is implemented and validated on the synthetic georeferenced
> fixture only. The real-data georeferenced path is NOT YET VERIFIED.**
> DFC2019 Track 1 tiles have no CRS or geotransform, so no DFC2019 DSM exists
> here. No external DEM (SRTM, CartoDEM or any other) was downloaded.

Phase 4a is frozen; Phase 4b only consumes its model and calibration code. The
DSM is assembled as

```
DSM(x) = T(x) + AGL(x) = T(x) + a·exp(z_rel(x)) + b
```

- **T** is terrain from a DEM reprojected onto the image grid. The DEM is
  **additive terrain only**: it is never a scale anchor for the relative
  field, and no DTM is subtracted.
- **Phase 6 remains cancelled**, because Phase 3 already predicts AGL (= nDSM).

```bash
python -m depthwizard.surfaces.run synthetic --config configs/phase4.yaml
```

Outputs are git-ignored, in `data/outputs/phase4b/`:
- `synthetic_dsm.tif` and `synthetic_dsm_known_calibration.tif`;
- `synthetic_dem_reprojected.tif` and `synthetic_agl.tif` (all four are COGs);
- the source DEM and the fixture;
- `phase4b_report.json`.

**Components:**
- `calibration/terrain.py` — `load_terrain_on_grid`.
  - The image grid is authoritative for CRS, transform, width, height and
    extent.
  - The DEM is reprojected with `rasterio.warp.reproject` and **bilinear**
    resampling.
  - Invalid pixels stay NaN / nodata, never 0. Invalid pixels are those outside
    the DEM, and those whose bilinear kernel touches a DEM nodata sample (where
    GDAL would otherwise renormalise the kernel, i.e. extrapolate; a test caught
    this).
  - It fails loudly (`TerrainError`) on:
    - a missing DEM or image CRS;
    - a missing (identity) or degenerate transform;
    - invalid dimensions;
    - an untransformable CRS pair;
    - no overlap;
    - no valid terrain pixel.
- `surfaces/dsm.py` — `calibrated_agl` and `fuse_dsm`.
  - Invalid terrain or invalid AGL gives a nodata DSM pixel.
  - It fails loudly (`DsmError`) on non-finite `a, b`, or on `a ≤ 0`, since a
    DSM export would then invert the relative field; override with
    `require_positive_scale=False`.
  - It also fails on an entirely invalid terrain or AGL grid.
- `ingest/geotiff.py` — `write_cog` and `validate_cog`.
  - Writing goes through GDAL's COG driver: DEFLATE with the floating-point
    predictor (lossless), 256 px tiles, automatic averaged overviews, float32,
    nodata −9999.
  - Validation reopens the file and checks: COG layout, tiling, overviews,
    CRS, transform, dimensions, bounds, nodata, dtype, readability, and the
    pixel values against memory. It raises `CogError` on any mismatch.
- `ingest/synthetic.py` — the fixture extension.
  - `TerrainSpec` is a deterministic tilted plane in the image CRS (540 m base,
    slopes +0.02 east and −0.015 north).
  - `write_synthetic_dem` samples it on a 1 arc-second **EPSG:4326** grid, so
    reprojection is always exercised.
  - A plane is used because bilinear resampling reproduces it exactly, so any
    residual is georeferencing error.

**SYNTHETIC acceptance run** (deterministic):

| item | value |
| --- | --- |
| image grid | EPSG:32643, 512 × 512, 0.5 m, origin (700000, 3170000) |
| DEM | EPSG:4326, 17 × 16 px at 1″ |
| reprojection | EPSG:4326 → EPSG:32643, bilinear, onto 512 × 512 |
| terrain vs analytic truth | max 3.6 × 10⁻⁵ m, MAE 9.8 × 10⁻⁶ m, RMSE 1.2 × 10⁻⁵ m |
| relative field | `z_rel = log(AGL_true + 1) + 0.5` in float32, the Phase 4a synthetic stand-in; the grey fixture says nothing through a real Phase 3 network |

DSM vs `T_true + AGL_true`, over 262,144 valid pixels with 0 nodata:

| DSM | a, b | MAE | RMSE | max abs |
| --- | --- | ---: | ---: | ---: |
| `synthetic_dsm_known_calibration.tif` (known a, b: **geometric acceptance**) | 0.6065, −1 | 1.8 × 10⁻⁵ m | 2.1 × 10⁻⁵ m | **6.6 × 10⁻⁵ m** (tolerance 1 mm: **PASS**) |
| `synthetic_dsm.tif` (a, b from the frozen Phase 4a synthetic calibration) | 0.6124, −1.329 | 0.314 m | 0.316 m | 0.319 m |

- The 1 mm tolerance is well above float32 storage (~3 × 10⁻⁵ m at 540 m) and
  well below the 5 mm a half-pixel georeferencing error would cause on this
  slope.
- The Phase 4a-calibrated DSM's error is exactly the Phase 4a synthetic
  calibration error (b off by −0.33 m). No new error comes from the DSM stage.

**COG validation:** all four products passed. Each has COG layout, 256 × 256
tiles, a 2× overview, DEFLATE, and the CRS, transform, dimensions, bounds,
nodata and dtype preserved. The pixels match memory exactly.

**QGIS:** not performed (deferred).

**Not yet verified:**
- a real georeferenced scene with a real DEM;
- DFC2019, which has no georeferencing.

### Phase 5 — validation harness ✅ IMPLEMENTED

> **Real-world validation: measured once, as a city-held-out model evaluation
> with LiDAR-anchored per-tile metric calibration (JAX → OMA). This is NOT an
> unseen-city zero-shot result.**
> - The Phase 3 model was trained on JAX only. OMA was held out of both
>   training and checkpoint selection, and both leakage audits passed.
> - OMA lidar sets each tile's metric scale through 2–3 reference buildings,
>   which are excluded from scoring.
> - Only 714 of 7,072 OMA tiles (10 %) could be calibrated, and the pooled
>   metrics are dominated by that per-tile calibration.
>
> See [the result](#real-world-validation-result-jax--oma) and
> [how to reproduce it](#reproducing-the-jax--oma-experiment). The shipped
> `best.pt` (trained on both cities) is still refused by the audit.

Code lives in `depthwizard.validation`; config in `configs/phase5.yaml`.
Commands:

```bash
# SYNTHETIC VALIDATION: Phase 4b products vs the Phase 0 fixture's specified truth
# (needs: python -m depthwizard.surfaces.run synthetic --config configs/phase4.yaml)
python -m depthwizard.validation.run synthetic --config configs/phase5.yaml

# REAL-WORLD VALIDATION: Phase 4a calibrated AGL on DFC2019 vs DFC2019 lidar AGL,
# on a geographically held-out city
python -m depthwizard.validation.run dfc --config configs/phase5.yaml \
    --relative-config configs/phase3.local.yaml
```

Both targets run the same steps:
1. Load the config and identify the held-out region.
2. Load the predictions and references.
3. Build the validity mask and the terrain/building masks.
4. Compute the metrics.
5. Write the error raster and figure.
6. Compute the confidence proxy.
7. Write the Markdown and JSON report, and print a short summary.

The harness never trains, fits or recalibrates anything. It never swaps one
data source for another: the `dfc` target does not fall back to synthetic data.
A missing input (a Phase 4b/4a report, a checkpoint, the DFC2019 root, a CLS
raster) fails with a message that names it. The CLI then exits with status 2.

#### SYNTHETIC VALIDATION vs REAL-WORLD VALIDATION

| | SYNTHETIC VALIDATION (`synthetic`) | REAL-WORLD VALIDATION (`dfc`) |
| --- | --- | --- |
| prediction | Phase 4b `synthetic_agl.tif` and `synthetic_dsm.tif` | Phase 4a `a·exp(z_rel) + b` per 512 px tile; `z_rel` re-inferred with the checkpoint the Phase 4a report names, `a, b` read from its `per_tile.jsonl` |
| reference | the fixture's **specified** heights (DSM: + the analytic terrain plane) | DFC2019 lidar AGL (metres) |
| terrain / building | the fixture's specified footprints | CLS 2 = terrain, CLS 6 = building |
| what it shows | the harness and the pipeline's geometry are correct | accuracy on a held-out city, with LiDAR-anchored per-tile calibration (not zero-shot) |
| status | run | **measured** for JAX → OMA (`epoch_009.pt`); **refused** for the shipped `best.pt` |

Synthetic metrics are never real-world accuracy. Every synthetic report and
figure is labelled `SYNTHETIC`.

#### Spatial validation rule

Training and held-out regions must be **geographically separate**:
- The regions are cities, the only region identifier DFC2019 carries
  (`JAX_004_007` → `JAX`).
- `configs/phase5.yaml` declares `train_regions: [JAX]` and
  `heldout_regions: [OMA]`. OMA is held out because Phase 4a returns a shadow
  calibration almost only on OMA tiles.
- There is no random-split option.

The declaration is **checked against the checkpoint itself**. Every Phase 3
checkpoint embeds its config, so `spatial_split.audit_region_split` recomputes
exactly which pairs it was trained on. The run is refused (`SplitError`) when:
- a held-out city appears in training;
- the checkpoint used a random split;
- the checkpoint trained on an undeclared region, or a declared training region
  never trained;
- any geographic tile falls on both sides.

The audit also refuses **model-selection leakage**. A checkpoint can be
trained without the held-out city and still have been *chosen* with it. When
the held-out city is the checkpoint's validation side, `best.pt` is the epoch
with the lowest held-out validation loss. In that case only the **final
pre-set epoch** (`training.epochs − 1`, i.e. `epoch_009.pt` for 10 epochs) is
accepted. The check reads only metadata every Phase 3 checkpoint stores
(`epoch`, `best_epoch`, `training.epochs`); a checkpoint without it is
refused, because its selection cannot be verified.

One caveat the audit cannot remove: Phase 4a fixes each tile's sun scale from
the DFC2019 lidar AGL of 2–3 reference buildings **inside that tile**. Phase 5
excludes those buildings' pixels from every metric, and the report says so.
Still, a held-out-city result is not entirely truth-free: that ground truth
enters the calibration.

The JAX → OMA experiment is therefore a **city-held-out model evaluation with
LiDAR-anchored per-tile metric calibration**. It is **not** an "unseen-city
zero-shot" result:
- the model (Phase 3) was trained on JAX only and never saw OMA, in training or
  in model selection;
- OMA lidar is used in two separate roles: per-tile calibration, through the
  2–3 reference buildings per tile, and independent scoring, with those
  reference buildings excluded.

**The shipped `best.pt` is still refused.** It was trained with
`per_city_scene`, i.e. tile-disjoint within **both** cities, so the audit finds
OMA pairs among its training pairs. Its refusal report is
`data/outputs/phase5/dfc/validation_report.md` (OMA: 1,397 training pairs).

The experiment that satisfies the rule was run in four steps, in this order.
Each is detailed [below](#real-world-validation-result-jax--oma).
1. Train a Phase 3 checkpoint on JAX only
   (`split: {mode: scene_prefix, val_scene_prefixes: [OMA]}`). Use its
   **final epoch** (`epoch_009.pt`), **not** `best.pt`: with this split,
   `best.pt` is chosen by OMA validation loss, and Phase 5 refuses it.
2. **Before any OMA result was inspected**, re-check the Phase 4a thresholds on
   the JAX-only training split. The frozen thresholds in `configs/phase4.yaml`
   came from an ablation whose training split included OMA tiles.
3. Run the held-out Phase 4a calibration on OMA with that checkpoint.
4. Run the Phase 5 `dfc` command on that Phase 4a output.

#### Metrics

Metrics use valid pixels only, where `e = prediction − reference`:

| metric | definition | units |
| --- | --- | --- |
| MAE | mean(\|e\|) | the product's units (read from its `UNITS` tag; never assumed) |
| RMSE | `sqrt(mean(e²))` | same |
| bias | `mean(e)` (positive = over-prediction) | same |
| Pearson r | correlation of prediction with reference | dimensionless |
| δ < 1.25, δ < 1.25², δ < 1.25³ | fraction of pixels with `max(p/r, r/p) < t` | fraction in [0, 1] |

- **Pearson r** is *not computable* in three cases: no valid pixels, fewer
  than 3 pixels, or zero variance in the prediction or the reference. The
  report then gives that reason, never 0.
- **δ** is evaluated only where prediction **and** reference are both > 0. The
  pixels excluded as non-positive are counted and reported. For AGL this
  excludes flat ground (reference 0 m), so terrain δ on AGL is not computable
  by definition.
- Metrics are pooled across tiles from additive sums, so any number of tiles
  streams exactly on the CPU.

#### NaN / nodata handling and valid-pixel counting

`masks.build_valid_mask` returns the evaluation mask **and** a count of every
excluded pixel. Each pixel is counted once, under the first reason that
applies:
1. prediction nodata;
2. prediction NaN/inf;
3. reference nodata;
4. reference NaN/inf;
5. reference marked invalid;
6. outside the region;
7. explicitly excluded (on DFC2019, the reference buildings each tile's sun
   scale was fitted on).

Valid plus excluded always equals the total. The metric accumulator refuses a
mask that lets a non-finite value through, so nothing is dropped out of sight.
The report lists the valid-pixel count for every product and category.

#### Terrain and building metrics

Every metric is reported separately for **OVERALL**, **TERRAIN** and
**BUILDING**. The masks come from labels the data carries: DFC2019 CLS, or
the fixture's specified footprints. They are never derived from the prediction
or the reference height. On DFC2019, other CLS classes (vegetation, water,
bridges, unlabelled) count only towards OVERALL. A category the data cannot
supply is reported as not yet measured, never as zero.

#### Error map

`error = prediction − reference` per pixel, on the prediction's own grid.
Invalid pixels are nodata (−9999, never 0).

- **GeoTIFF**:
  - Georeferenced grids are written as COGs through the Phase 4b
    `write_cog` / `validate_cog`, with CRS, transform, size, bounds, nodata and
    every pixel checked on read-back. They open in QGIS as-is.
  - DFC2019 tiles have no CRS, so their error maps are written without one,
    keeping the source's pixel-space transform. Nothing is invented to look
    georeferenced.
- **PNG**:
  - A diverging colour map centred on zero (blue = under-prediction, red =
    over-prediction), with invalid pixels in grey and map-coordinate axes when
    georeferenced.
  - Beside the map is a histogram of **all** valid errors, unclipped.
  - The colour scale spans ± the 99th percentile of |error|. Larger errors are
    drawn in the end colours, never hidden; they are counted, and the true
    min/max are printed. The raster itself is never clipped.

#### Confidence proxy

Rule-based flags, **not** a learned uncertainty and **not** a probability of
correctness. They use only what earlier phases produce:

| condition | source | effect |
| --- | --- | --- |
| shadow occlusion | Phase 2 `ClassicalShadowDetector` (unchanged) | REDUCED |
| water | a data label (DFC2019 CLS 9) | UNSUITABLE |
| sun elevation outside 25°–45° (inclusive) | Phase 2 `sun_band_status` on the scene's sun elevation | REDUCED (every pixel of the scene) |

- **States**, with precedence (first applicable wins):
  - `INVALID`;
  - `UNSUITABLE`;
  - `REDUCED`;
  - `NOT_ASSESSED`: no flag fired, but an input was unavailable;
  - `HIGH`: every check ran and none fired.
- An unavailable input is reported as "not available", never as "0 pixels
  affected". Pixels it would cover cannot be HIGH.
- On the fixture, water is not available (no land-cover labels).
- On DFC2019, the sun band is not available: there is no sun metadata, and the
  GSD is unverified.
- On DFC2019 the shadow detector also marks dark asphalt (a Phase 4a finding),
  so the shadow flag over-counts there.
- The report also gives MAE/RMSE per confidence state, so you can check
  whether the proxy separates good pixels from bad ones.

#### Generated artifacts (git-ignored)

| path | contents |
| --- | --- |
| `data/outputs/phase5/synthetic/validation_report.md` | Markdown report, SYNTHETIC |
| `data/outputs/phase5/synthetic/phase5_report.json` | the same, machine-readable |
| `data/outputs/phase5/synthetic/error_map_{agl,dsm}.tif` | error COGs, EPSG:32643 |
| `data/outputs/phase5/synthetic/error_map_{agl,dsm}.png` | error figures |
| `data/outputs/phase5/synthetic/confidence_{agl,dsm}.tif` | confidence-state rasters (codes 0–4) |
| `data/outputs/phase5/dfc/validation_report.md`, `phase5_report.json` | the refusal of the shipped `best.pt` (trained on both cities), with "not yet measured" |
| `data/outputs/phase3_jax_only/` | JAX-only Phase 3: `checkpoints/epoch_000…009.pt` (plus `best.pt`, **not used**), `train.log`, `phase3_jax_only_report.json` |
| `data/outputs/phase4a_jax_only_training_ablation/` | JAX-only ablation: `ablation_report.json`, `decision_analysis.json` (bootstrap), `frozen_decisions.json`, `buildings.jsonl`, `tiles.jsonl`, `run.log` |
| `data/outputs/phase4a_jax_only/` | held-out OMA Phase 4a: `phase4a_report.json`, `per_tile.jsonl`, 4 diagnostic figures, `run.log` |
| `data/outputs/phase5_jax_only/dfc/validation_report.md`, `phase5_report.json` | the REAL-WORLD Phase 5 report (JAX → OMA) |
| `data/outputs/phase5_jax_only/dfc/error_maps/<tile>.tif` | 714 per-tile error GeoTIFFs (no CRS, 512 × 512, float32, nodata −9999) |
| `data/outputs/phase5_jax_only/dfc/error_maps/<tile>.png` | 8 error figures: the first tiles in sorted tile-ID order, never chosen by accuracy |

#### SYNTHETIC VALIDATION result (specified, not measured, heights)

This is the fixture run through the harness. It checks the harness, not
real-world accuracy. The relative field is the Phase 4a/4b constructed
stand-in, not a Phase 3 prediction.

| AGL | valid px | MAE (m) | RMSE (m) | Pearson r | δ < 1.25 |
| --- | ---: | ---: | ---: | --- | --- |
| overall | 262,144 | 0.3140 | 0.3156 | 1.0000 | 1.0000 |
| terrain | 253,844 | 0.3190 | 0.3190 | not computable (constant field) | not computable (reference 0 m) |
| building | 8,300 | 0.1620 | 0.1832 | 1.0000 | 1.0000 |

- The DSM gives the same MAE/RMSE. They equal the figures Phase 4b recorded
  for `synthetic_dsm.tif`, and a test asserts this.
- The shadow flag marks 12,599 pixels (4.8 %) REDUCED.
- The sun (45°) is within the band.
- Water is not available, so no pixel is HIGH.

#### REAL-WORLD VALIDATION result: JAX → OMA

> **City-held-out model evaluation with LiDAR-anchored per-tile metric
> calibration. NOT an unseen-city zero-shot result.**
> - Training region: **JAX**. Held-out region: **OMA**, excluded from Phase 3
>   training and checkpoint selection.
> - OMA lidar sets each tile's metric scale (2–3 reference buildings per tile).
>   Those buildings are excluded from scoring.
> - The pooled figures below cover 10 % of OMA tiles and are dominated by
>   per-tile calibration error. They are **not** a measure of the Phase 3
>   model's quality on its own.

**Step 1 — JAX-only Phase 3 training.** The shipped `configs/phase3.yaml`
settings were used unchanged, apart from the split and the output paths.
- Split: `scene_prefix`, `val_scene_prefixes: [OMA]`.
- Training: 1,015 JAX pairs (53 geographic tiles). Validation side: 1,768 OMA
  pairs (55 geographic tiles). No scene overlap.
- 10 epochs × 508 optimiser steps, bf16 on the RTX 4050, peak VRAM 297 MiB,
  about 15 min 45 s wall-clock. No OOM, no NaN/Inf, 0 skipped batches.
- **Downstream checkpoint: `epoch_009.pt`**, the final pre-set epoch
  (train loss 0.34271, OMA validation loss 0.40826).
  - `best.pt` is epoch 2, selected by OMA validation loss (0.39193). It was
    **not used**, and Phase 5 refuses it.
  - Validation loss was monitored only: the learning-rate schedule is
    epoch-based cosine, so it never influenced the weights.
- The original `data/outputs/phase3/` checkpoints were verified unchanged
  (SHA-256).

**Step 2 — JAX-only Phase 4a training-split ablation** (no OMA data read).
- Data: 209 JAX training images (4 views per geographic tile), 53 geographic
  tiles, 836 tiles of 512 px. The frozen variant has 18 calibratable tiles and
  160 leave-references-out targets.
- The existing `depthwizard.calibration.ablation` functions were used
  unchanged. Unlike `run ablation`, the run skipped the held-out height census
  (which would read OMA lidar); see
  [Reproducing](#reproducing-the-jax--oma-experiment).
- **Decision rule, fixed before the run:**
  - criteria: within-tile Spearman of L vs h, MAD of log k, and
    leave-references-out (LRO) MAE;
  - uncertainty: a 95 % bootstrap over geographic tiles (1,000 draws, seed
    20260928);
  - **CHANGE** only if an alternative is significantly better on at least one
    criterion and significantly worse on none;
  - a better point estimate whose interval includes 0 is **INSUFFICIENT
    EVIDENCE** (the frozen choice is kept);
  - otherwise **KEEP**.

Frozen variant: Spearman 0.209, MAD log k 0.261, LRO MAE 2.53 m. Differences
are alternative − frozen, with 95 % intervals.

| setting | frozen | JAX-only evidence | decision |
| --- | --- | --- | --- |
| gap tolerance | 2 px | LRO MAE: 4 px −0.64 m [−1.74, +0.41]; 6 px −0.69 m [−2.09, +0.02]. Validity is not better at either. | INSUFFICIENT EVIDENCE → keep |
| isolation | 0 (off) | The rule literally gave CHANGE: 10 px LRO MAE −1.21 m [−2.06, −0.41]. But that comparison is unpaired: 10 px calibrates 3 tiles, all from one geographic tile (JAX_022). On the same tiles: MAE 1.322 vs 1.393 m, and median error worse (1.177 vs 1.121 m). | INSUFFICIENT EVIDENCE → keep (owner's decision) |
| merged-shadow check | on | Turning it off: LRO MAE +0.33 m [0.00, +1.15]. It is worse on every point estimate. | KEEP |
| shadow threshold | image Otsu | Ground-only Otsu: LRO MAE −0.31 m [−1.63, +0.38]. Nothing significant. | INSUFFICIENT EVIDENCE → keep |
| k estimator | median | Mean: +0.04 m. Inverse-variance: −0.22 m [−0.59, +0.08]. | KEEP / INSUFFICIENT EVIDENCE → keep |
| reference gates, footprint, azimuth and fit settings, a/SE ≥ 2 label | as shipped | Never ablated (not data-derived). | KEEP |

No setting changed: `configs/phase4.yaml` is untouched. The decisions are
recorded in
`data/outputs/phase4a_jax_only_training_ablation/frozen_decisions.json`.

**Step 3 — held-out OMA Phase 4a calibration** (`epoch_009.pt`, frozen
thresholds; 1,764 s).
- Data: all 1,768 OMA images (55 geographic tiles), 7,072 tiles of 512 px.
  No JAX tile was processed.

| stage | tiles |
| --- | ---: |
| all OMA tiles | 7,072 |
| with buildings | 4,898 |
| sun azimuth recovered | 5,200 |
| ≥ 2 valid references (sun calibrated) | 831 |
| a, b fit returned | **714** (444 images, 26 of 55 geographic tiles) |
| fit `identifiable_positive` | **166** (20 % of 831 attempted) |

- Of the 714 fits: 166 `identifiable_positive`, 267 `statistically_weak`,
  281 `non_positive`. Another 117 attempts were unidentifiable (insufficient
  constraints).
- The fitted scale is mostly unidentifiable: median a/SE = 0.54.
- **No tile with buildings ≥ 20 m could be calibrated** (56 tiles, 0 fits).
- Per-building shadow evaluation (no a, b): 11,029 evaluated / 17,864
  eligible / 62,510 kept footprints. MAE 2.77 m, RMSE 4.96 m, Spearman of
  shadow height vs true height 0.16.
- ORACLE control (diagnostic only; a, b fitted to lidar): building-pixel MAE
  1.58 m, vs 2.82 m with the shadow a, b. The JAX-trained field does carry
  height structure in OMA; the shadow calibration adds about 1.2 m.

**Step 4 — Phase 5 validation.**
- Command:
  `python -m depthwizard.validation.run dfc --config configs/phase5.jaxonly.local.yaml --relative-config configs/phase3.local.yaml`.
  Exit 0.
- **Spatial split audit: PASSED.**
  - Checkpoint split `scene_prefix`; training pairs `{JAX: 1015}`.
  - Validation side `{OMA}`: 1,768 pairs over 55 geographic tiles, 0 scene
    overlap, not random.
- **Model-selection audit: PASSED.** Epoch 9 of 10 (the final epoch;
  `best_epoch` 2).
- Evaluated: the 714 calibrated tiles from 444 images. All three fit labels
  are included, as `configs/phase5.yaml` fixed before any OMA result existed.
- Valid pixels: 185,197,749 of 187,170,816.
  - 1,973,064 calibration reference-building pixels were excluded.
  - 3 non-finite reference pixels were excluded.

AGL, metres; δ values are fractions over pixels where prediction and reference
are both > 0:

| region | valid px | MAE | RMSE | bias | Pearson r | δ < 1.25 | δ < 1.25² | δ < 1.25³ | δ domain px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 185,197,749 | 3.9986 | 5.9945 | +1.4502 | 0.0594 | 0.1171 | 0.2252 | 0.3206 | 117,740,159 |
| terrain (CLS 2) | 123,170,402 | 3.8606 | 5.6801 | +3.1151 | 0.0160 | 0.0491 | 0.0968 | 0.1423 | 60,490,946 |
| building (CLS 6) | 21,597,325 | 2.8245 | 4.1555 | −1.0581 | 0.0884 | 0.2439 | 0.4520 | 0.6153 | 20,908,212 |

Overall and building MAE/RMSE, and the valid-pixel count, match Phase 4a's
independent computation on the same tiles.

Confidence proxy:

| check | status | flagged valid px |
| --- | --- | ---: |
| shadow occlusion | evaluated (Phase 2 detector) | 102,918,279 (55.6 %) |
| water | evaluated (CLS 9) | 785,413 (0.42 %) |
| sun band 25–45° | **not available** (no sun metadata; GSD unverified) | not available |

| state | pixels | MAE / RMSE (m) |
| --- | ---: | --- |
| UNSUITABLE | 785,413 | 4.36 / 5.98 |
| REDUCED | 102,376,703 | 4.13 / 6.28 |
| NOT_ASSESSED | 82,035,633 | 3.83 / 5.61 |
| HIGH | 0 | not yet measured |

Error maps: 714 GeoTIFFs, each written, read back and matched to memory
exactly. They have no CRS (DFC2019 has none), a pixel-space transform with the
tile offset, and nodata −9999. There are also 8 PNG figures; see
[Generated artifacts](#generated-artifacts-git-ignored).

**How to read these numbers.**
- **Pearson r ≈ 0.06 is not a model-quality result.** It pools 714 tiles, each
  with its own shadow-calibrated a, b, and most of those scales are
  unidentifiable. Per-tile offset error dominates.
- **The identifiable fits are not more accurate** (Phase 4a: all-pixel MAE
  2.95 m for `identifiable_positive` vs 2.82 m for `statistically_weak`).
- **Terrain δ is low largely because the terrain reference AGL is near 0 m**,
  where small absolute errors give large ratios.
- **The samples are not independent.** Several views of the same place are
  evaluated, and 26 of 55 OMA geographic tiles contribute.
- These figures are not comparable with the Phase 4a numbers above, which came
  from a different model and a different held-out set.

**Not yet measured:**
- the 6,358 uncalibrated OMA tiles (90 %) and the 29 uncalibrated geographic
  tiles;
- buildings ≥ 20 m (no such tile could be calibrated);
- HIGH-confidence pixels, and the sun-band check (not possible on DFC2019);
- a georeferenced error mosaic (DFC2019 Track 1 has no georeferencing);
- any Jacksonville performance;
- any truth-free (zero-shot) metric result.

#### Reproducing the JAX → OMA experiment

The experiment uses three **git-ignored** local configs (`configs/*.local.yaml`)
that are not in the repository. Each is a small, documented change to a
tracked config. `configs/phase3.local.yaml` is the existing machine-specific
file that points at your DFC2019 copy.

`configs/phase3.jaxonly.local.yaml`: everything else is inherited from
`phase3.local.yaml` → `phase3.yaml`.

```yaml
extends: phase3.local.yaml
run_id: phase3_jax_only
output_dir: data/outputs/phase3_jax_only
dataset:
  split:
    mode: scene_prefix
    val_scene_prefixes: [OMA]
checkpoint:
  dir: data/outputs/phase3_jax_only/checkpoints
```

`configs/phase4.jaxonly.local.yaml`: a copy of `configs/phase4.yaml` with
**exactly four keys changed** and every threshold identical. (The Phase 4 loader
has no `extends`.)

```bash
sed -e 's|^run_id: phase4a$|run_id: phase4a_jax_only|' \
    -e 's|^output_dir: data/outputs/phase4a$|output_dir: data/outputs/phase4a_jax_only|' \
    -e 's|^relative_config: configs/phase3.yaml$|relative_config: configs/phase3.jaxonly.local.yaml|' \
    -e 's|^checkpoint: data/outputs/phase3/checkpoints/best.pt$|checkpoint: data/outputs/phase3_jax_only/checkpoints/epoch_009.pt|' \
    configs/phase4.yaml > configs/phase4.jaxonly.local.yaml
```

`configs/phase5.jaxonly.local.yaml`: a copy of `configs/phase5.yaml` with
**exactly two keys changed**. The split (train JAX, hold out OMA) and
`fit_classes` are unchanged.

```bash
sed -e 's|^output_dir: data/outputs/phase5$|output_dir: data/outputs/phase5_jax_only|' \
    -e 's|^  phase4a_dir: data/outputs/phase4a$|  phase4a_dir: data/outputs/phase4a_jax_only|' \
    configs/phase5.yaml > configs/phase5.jaxonly.local.yaml
```

Commands, in order:

```bash
# 1. JAX-only Phase 3 (use epoch_009.pt downstream, never best.pt)
python -m depthwizard.relative.train --config configs/phase3.jaxonly.local.yaml

# 2. JAX-only Phase 4a training-split ablation (see note below). --output-dir keeps
#    a rerun out of the recorded data/outputs/phase4a_jax_only_training_ablation/
#    (the ablation writes to <output-dir>_training_ablation)
python -m depthwizard.calibration.run ablation --config configs/phase4.jaxonly.local.yaml \
    --relative-config configs/phase3.jaxonly.local.yaml \
    --output-dir data/outputs/phase4a_jax_only_rerun

# 3. held-out OMA Phase 4a calibration
python -m depthwizard.calibration.run dfc --config configs/phase4.jaxonly.local.yaml \
    --relative-config configs/phase3.jaxonly.local.yaml

# 4. Phase 5 real validation (the relative config only supplies the DFC2019 root;
#    the split is read from the checkpoint)
python -m depthwizard.validation.run dfc --config configs/phase5.jaxonly.local.yaml \
    --relative-config configs/phase3.local.yaml
```

**Note on step 2.** The recorded ablation was not run through this CLI. It used
an untracked driver that calls the same unmodified functions
(`select_views(train, 4)`, `building_gate_records`, `summarise_ablation`,
`height_census` on the training side). The driver differs in only two ways:
- it asserted the training side is JAX only;
- it skipped `run_ablation`'s held-out height census, which would read OMA
  lidar.

The CLI above reproduces the same per-building measurements and summary into
`data/outputs/phase4a_jax_only_rerun_training_ablation/`, but its report also
contains an OMA height census. That census is a data description, never used
for any decision.

The bootstrap decision analysis (`decision_analysis.json`) was also an
untracked script. Its method is fully specified above, and it re-uses
`summarise_variant`:
- resample the geographic tiles with replacement;
- recompute each variant's criteria;
- take the 2.5 / 97.5 percentiles of the alternative − frozen differences
  (1,000 draws, seed 20260928).

#### Tests

Tests are `tests/test_validation_{metrics,split_and_confidence,error_map_report,run}.py`.
They cover:
- the metric and mask edge cases;
- split separation, leakage, and random-split refusal;
- model-selection leakage: a best-validation-loss or other non-final epoch is
  refused when the held-out city is the checkpoint's validation side, and the
  final epoch is accepted;
- the sun band at and around 25° and 45°, shadow, water and combined
  conditions;
- CRS/transform/nodata preservation;
- report rendering, including that an empty result renders no numbers;
- the full synthetic pipeline;
- a DFC2019-shaped run with a known 0.5 m error, and refused leaked or
  best-epoch checkpoints;
- the measured-report wording, and that a refused report doesn't claim it.

The DFC2019-shaped tests use synthetic rasters and an injected predictor. The
real checkpoint-inference path has since run on the 714 calibrated OMA tiles
in the acceptance run above.

The Phase 2 reference interface (`ReferenceSet`, JSON/CSV loading, scatter
plot) is unchanged. No manually measured real references have been supplied.

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
