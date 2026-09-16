# DepthWizard

**Building height estimation from single-view satellite imagery, using shadow physics.**

Smart India Hackathon 2026 — problem statement **SIH26175**.

> **Status: Phase 0 (scaffolding) only.**
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

Cross-cutting, used by every stage: `config.py` (YAML + dataclasses) and
`logging_setup.py` (structured logging).

Of the above, **only `ingest` and `physics` have any implementation.**
`calibration`, `surfaces`, `validation` and `viewer` are empty placeholders
that exist so import paths stay stable from the first commit.

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
│   └── make_synthetic_fixture.py # CLI entry point
├── src/depthwizard/
│   ├── config.py                 # YAML -> validated dataclasses
│   ├── logging_setup.py          # text / JSON structured logging
│   ├── ingest/
│   │   ├── geotiff.py            # GeoTIFF read + write helpers
│   │   └── synthetic.py          # synthetic fixture generator
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

### Verify a generated GeoTIFF

```bash
python -c "
from depthwizard.ingest.geotiff import read_raster, read_sun_metadata
r = read_raster('data/processed/synthetic/synthetic_city_a.tif')
print('crs      ', r.crs)
print('gsd_m    ', r.gsd_m)
print('north-up ', r.is_north_up)
print('transform', r.transform)
print('sun      ', read_sun_metadata('data/processed/synthetic/synthetic_city_a.tif'))
"
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

### Phase 1 — shadow extraction ❌ NOT IMPLEMENTED

- [ ] Real satellite imagery ingest and metadata parsing
- [ ] Shadow segmentation from imagery
- [ ] Building footprint association
- [ ] Shadow length measurement along the solar azimuth

### Phase 2 — height estimation & calibration ❌ NOT IMPLEMENTED

- [ ] Height estimation from measured shadows
- [ ] Terrain slope and off-nadir view corrections
- [ ] Per-scene bias calibration
- [ ] Learned refinement model *(no model is built in Phase 0)*

### Phase 3 — surface products ❌ NOT IMPLEMENTED

- [ ] DSM generation
- [ ] DTM extraction
- [ ] nDSM (normalised height surface)

### Phase 4 — validation ❌ NOT IMPLEMENTED

- [ ] Scoring against the synthetic fixture's ground truth
- [ ] Scoring against real reference DSM / lidar
- [ ] Error reporting *(no metrics are claimed until they are measured)*

### Phase 5 — viewer ❌ NOT IMPLEMENTED

- [ ] FastAPI service
- [ ] Web map viewer and per-building inspection

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
