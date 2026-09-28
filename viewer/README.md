# DepthWizard viewer (Phase 7, complete and accepted)

A browser viewer for the pipeline's height products, built with React, plain
Three.js and Vite. No CesiumJS, no react-three-fiber.

**Status: Phase 7 Steps 2–7 are implemented.** Phase 8 is **not** started. No uncertainty is displayed: the
pipeline has no per-pixel uncertainty product ("Uncertainty: not yet measured").

## Run

```bash
cd viewer
npm install        # first time only
npm run dev        # http://localhost:5173
npm test           # vitest (Node; real-data tests skip when data/outputs is absent)
```

The dev server serves `../data/outputs` read-only at `/data/outputs/*` and lists
the pipeline reports at `/__data_index` (`server/dataRoute.js`). Run Phases 4b,
5 and 6 first so their outputs exist.

## What it does

| Step | Module(s) | What |
|---|---|---|
| 2 | `src/data/*` | Loads the Phase 4b/5/6 products by following the reports' own links (no hard-coded paths); GeoTIFF/COG via geotiff.js, Phase 3 `.npy` via a small reader. Nodata stays nodata. |
| 3 | `src/3d/heightmap.js`, `quadtree.js`, `tileMesh.js`, `terrain.js` | Height raster → quadtree-LOD indexed mesh; crack-free seams; nodata never bridged. |
| 4 | `src/3d/texture.js`, `cameraRig.js` | Same-grid source image as texture (grayscale stays grayscale); orbit controls. |
| 5 | `src/3d/firstPerson.js`, `measure.js` | First-person flythrough; click-to-measure via raycast → nDSM. |
| 6 | `src/3d/overlays.js`, `linkedCursor.js` | Phase 5 error / confidence overlays (only where they describe the displayed surface); 2D ↔ 3D linked cursor. |
| 7 | `src/export/gltfExport.js`, `png.js`, `dracoEncoder.js` | Displayed raster → `.glb` (full resolution, source-image texture, metadata), optionally Draco-compressed and verified. |

## Navigation

The toolbar shows **Camera: Orbit** or **Camera: First Person**.

- **Orbit** (default): left-drag rotate, right-drag pan, wheel zoom. Distance
  limits come from the terrain's size (closest 0.5 % of its diagonal, farthest
  3×); the camera cannot go below the horizon (polar ≤ 85°) and the orbit point
  stays over the terrain.
- **First person**: `W`/`S` forward/back, `A`/`D` strafe, `Q`/`E` down/up,
  `Shift` ×4 speed, drag the mouse to look around (no pointer lock). Eye height
  (0.5 % of the terrain diagonal) and speed (4 % per second) come from the
  loaded terrain. The camera stays inside the terrain's bounding box and never
  lower than the highest valid raster value within a small radius plus the eye
  height, so it rises over buildings instead of passing through them; over
  nodata it keeps the last known ground.
- **Reset view**: orbit → the initial overview; first person → standing at the
  overview's centre, facing the same way.

Both modes move the same camera and update the same quadtree LOD (at most once
per animation frame, re-using cached tiles).

## Click-to-measure

Press **Measure**, then click the terrain (a drag still navigates). Press it
again to exit; the marker and result are cleared.

1. A `THREE.Raycaster` ray from the click (canvas-relative coordinates) hits the
   rendered tiles.
2. Only the hit's **horizontal** position is used: world → map coordinate
   (undoing the viewer's re-centring) → the raster pixel containing it
   (pixel `(row, col)` covers `[col, col+1) × [row, row+1)`; floor, as
   `rasterio.rowcol`).
3. The value is **read from the raster** at that pixel and shown with its raw
   float32 value. The rendered mesh height is never reported — on a coarse LOD
   tile it is an approximation, the raster value is not.

What is measured:

- **Phase 6 data**: always the **nDSM** (height above ground), whichever surface
  is displayed. When the DSM or DTM is displayed, its own value at the same
  pixel is shown separately and labelled (e.g. DSM = absolute elevation).
- **Phase 3 `.npy`**: there is no nDSM; the relative raster itself is measured
  and reported **unitless** — never in metres.

The panel shows the value, product, raster row/column, map coordinate and CRS
(when georeferenced), mode (ABSOLUTE/RELATIVE), units, the LOD tile hit and the
response time of that click.

**No valid height at this location** is shown — with the reason — when the
click misses the terrain, falls outside the raster, lands on nodata or on a
non-finite value, or the measured raster cannot be related to the displayed
grid (e.g. a different CRS). No value is invented.

No uncertainty is computed or displayed in Step 5.

## Error and confidence overlays (Step 6)

**Error overlay** and **Confidence overlay** (ON/OFF, one at a time, adjustable
opacity) colour the surface. They change the material only: terrain heights,
geometry, LOD selection and the tile cache are untouched. The overlay is one
full-raster texture sampled through the same UVs as the source image, so it
follows whichever LOD tiles are selected, with no shift.

**What they are.** Phase 5 validated the **source DSM** (its `prediction_path`),
and Phase 6 re-exports that DSM as `dsm.tif` (tag `SOURCE_DSM`). So the
overlays are shown **only on the DSM surface** of the Phase 6 product whose
source DSM Phase 5 validated, and only when the diagnostic raster has exactly
the DSM's size, transform and CRS. Everywhere else the toggle stays empty and
the reason is shown, e.g.:

- nDSM shown: *"Error overlay unavailable for this nDSM — Phase 5 error map
  validates the source DSM, not nDSM."* and *"Confidence overlay unavailable
  for this nDSM — available only for the Phase 5 source DSM; not an nDSM
  confidence map."* (same for the DTM). A DSM diagnostic is never presented as
  an nDSM diagnostic.
- no Phase 5 report validated this product's source DSM (the current acceptance
  product): *"… no aligned error raster: no Phase 5 report validated …"*.
- misaligned grid, a DSM exported from a different source, an error map of
  another product, or a report/raster disagreement on the state codes.
- a Phase 3 relative `.npy`: no Phase 5 diagnostics.

**Error overlay:** *Phase 5 DSM error — prediction minus reference*. Diverging
blue–white–red, symmetric about 0: red = DSM above the reference
(over-prediction), blue = below, white = 0. The sign is never lost. Colour
saturates at ±L, where L is the 99th percentile of |error| over valid pixels,
shown in the legend together with the recorded min/max and pixel counts.

**Confidence overlay:** the Phase 5 **rule-based quality flags**
(`validation/confidence.py`): 0 INVALID, 1 UNSUITABLE, 2 REDUCED,
3 NOT_ASSESSED, 4 HIGH. The state table is read from the Phase 5 report and
cross-checked against the raster's `STATE_<code>` tags. The legend lists code,
state, the documented meaning and pixel counts, and states **"Rule-based quality
flags, not probabilities."** No percentage, probability or score is derived.

Nodata, non-finite values and undocumented codes are not drawn (the surface
shows through) and are counted. No replacement value is invented.

## 2D ↔ 3D linked cursor (Step 6)

**2D linked cursor** opens a small raster panel (the grayscale or RGB source
image as declared by the file, or the displayed surface's values on a grayscale
display ramp, plus the active overlay). Moving or clicking in the panel moves one
cyan marker in 3D. Hovering over the terrain (no button held) or clicking it
moves the 2D crosshair. The link is always a raster pixel of the displayed
surface:

- 2D point → continuous (col, row) → pixel (floor) → pixel centre
  (col + 0.5, row + 0.5) → map coordinate via the raster's affine transform →
  3D (minus the recentring origin, z = −north).
- 3D hit → map → the pixel containing it (the same `pixelFor` that
  click-to-measure uses).

The panel draws the raster grid as stored (row 0 at the top). This is pixel
space, not a map, so an arrow shows where map north points according to the
transform. Rotated, mirrored and non-square-pixel transforms are handled through
the transform, never assumed. The readout shows row/column, the map coordinate
of the pixel centre (with CRS; "not georeferenced" for pixel-space rasters), the
marker position, the raster value (or *nodata*), the overlay value when an
overlay is on, "Uncertainty: not yet measured", and the response time.

If the 2D product is not on the displayed grid, the link is disabled with the
reason. The marker is a single reused object that raycasts ignore. The cursor
does not change the camera mode, and it coexists with orbit, first person and
click-to-measure: one click can both measure and link, and both report the same
pixel.

## glTF export (Step 7)

Press **Export .glb** under the 3D view. It exports the **displayed** raster
(nDSM, DSM, DTM or a Phase 3 `.npy`) and downloads it. Tick or untick **Draco
compression** first. The status line shows *ready → exporting → exported
successfully / failed (reason)*. After an export it lists the measured file size,
vertex and triangle counts, and export time. A Draco-vs-uncompressed ratio is
shown only once both variants of the same raster have been exported. The button
is disabled when no height raster is loaded.

**What is in the file**

- One mesh at **full resolution** (every valid pixel), built with the viewer's
  own mesh code (`heightmap.js` + `tileMesh.js`, one node at stride 1). It is
  not the camera-dependent LOD tiles. For 512 × 512 that is 262,144 vertices and
  522,242 triangles.
- Positions: one vertex per valid **pixel centre** `(col + 0.5, row + 0.5)`
  through the raster's affine transform (rotation and non-square pixels
  included). glTF is Y-up: `+x` = east, `+y` = up, `−z` = north, re-centred on
  the raster centre for float32 precision. The origin is stored in the extras:
  `map x = x + origin[0]`, `map y = −z + origin[1]`.
- `NORMAL`, `TEXCOORD_0` (texel centres) and 32-bit indices.
- Texture: the source image, only when it is on the same grid, embedded
  losslessly as PNG. A grayscale image stays a 1-channel grayscale PNG.
  Otherwise the material is plain grey.
- Metadata in `asset.extras.depthwizard` (repeated on the node): source path,
  product, SYNTHETIC flag, width/height, CRS, transform, units, mode and reason,
  vertical scale applied vs displayed, coordinate frame, nodata value and
  counts, texture, compression settings.

**Heights.** `y` is always the raster value itself (vertical scale 1).

- **ABSOLUTE:** metres, 1:1 with the metre map axes.
- **RELATIVE:** the raw unitless values, labelled *"RELATIVE and unitless — NOT
  metres"*. The viewer draws RELATIVE rasters with a display-only
  exaggeration; that factor is recorded (`displayedVerticalScale`) but **not
  applied**. A `.npy` has pixel horizontal axes and no CRS.

**Nodata.** Nodata and non-finite pixels never become vertices. A triangle is
written only when all 3 vertices are valid, and holes stay open. Nothing is
filled or replaced.

**Not included:** the Phase 5 error/confidence overlays (diagnostics without
their legend), the measurement and linked-cursor markers, and uncertainty (no
such product exists).

**Draco.** Compression uses the glTF-standard `KHR_draco_mesh_compression`
extension (via `@gltf-transform/core` + `@gltf-transform/extensions`), encoded
with Google's official `draco3dgltf` WASM encoder. The encoder is served by Vite
from the package and loaded on first use. The panel shows "Draco encoder loaded"
or "Draco unavailable: …".

- If the encoder cannot load, a Draco export **fails** with that reason. It
  never writes an uncompressed file labelled Draco.
- Every written file is re-parsed. "Draco: verified in the written file"
  appears only when the extension is used, required, and present on the
  primitive.
- Draco is **lossy**: positions are quantised to 16 bits over the mesh's largest
  extent (normals 10, UVs 14). Uncompressed export is exact.
- Loading a Draco `.glb` needs a Draco decoder, e.g. three.js `DRACOLoader`.

## Phase 7 acceptance (2026-09-29)

**Status: Phase 7 (Steps 1–7) accepted. Phase 8 has not started.** Phase 0–6
code and configuration are unchanged.

- **Tests:** `npm test`: 16 files, **199 passed, 0 failed, 0 skipped** (the
  real-data tests ran because `data/outputs` was present).
- **Production build:** `vite build` succeeded; the dev-only debug handles are
  absent from the bundle.
- **Browser acceptance** (headless Chrome 1600 × 2400, SwiftShader WebGL, fresh
  Vite dev server): **75 / 75 checks passed** in four scripts (Step 6
  regression 36, Step 7 export 21, final extra checks 12, and click-to-measure
  on all 4 Phase 6 building centroids with and without the linked cursor 6).
  Each building's measured value equals rasterio and Phase 6 `height_at()`
  exactly, and the linked cursor reports the same pixel and value. No JS exceptions and
  no failed data requests; the only HTTP error was Chrome's automatic
  `/favicon.ico` 404. The scripts and their results are kept outside the
  repository.
- **Real data used** (all SYNTHETIC; no real-world validation is claimed):
  - `data/outputs/phase6/synthetic/phase6_report.json`: products
    `synthetic_dsm_known_calibration.tif` (acceptance input) and
    `synthetic_dsm.tif` (dsm/dtm/ndsm/ground_mask, 512 × 512, EPSG:32643, 0.5 m).
  - `data/outputs/phase5/synthetic/phase5_report.json`: `error_map_dsm.tif`,
    `confidence_dsm.tif` (they validate `phase4b/synthetic_dsm.tif`).
  - `data/outputs/phase4b/fixture/synthetic_city_a.tif`: grayscale source image
    (linked through the Phase 4b report).
  - `data/outputs/phase3/verification/JAX_004_006_r256_c256_relative_height.npy`
    (RELATIVE).
- **glTF / Draco:** every exported file was parsed (independent GLB parser and
  gltf-transform) and compared with rasterio/numpy.
  - Uncompressed: heights equal the source float32 values exactly.
  - Draco: `KHR_draco_mesh_compression` is used, required and present on the
    primitive (verified in the downloaded file). The largest height error at
    Phase 6 clicks + 400 samples was 0.0015 m (nDSM) and 0.0039 units
    (RELATIVE), within the 16-bit step.

**Measured this acceptance run** (headless Chrome, SwiftShader; single runs):

| What | Measured |
|---|---|
| Phase 6 load + decode (in-app) | 175 ms (button → mesh shown: 553 ms wall) |
| First mesh | heightmap 1.6 ms, quadtree + texture 0.9 ms, first LOD + render 32.5 ms |
| Mode switch | orbit → first person 193.0 ms; first person → orbit 2.7 ms |
| Overlay activation (in-app) | error 253.1 ms (first build), confidence 46.3 ms |
| Linked cursor | 2D → 3D 0.7 ms; 3D click → 2D 3.2 ms (raycast + lookup) |
| Export nDSM uncompressed | 667.8 ms, 14,663,040 B, 262,144 vertices, 522,242 triangles |
| Export nDSM Draco | 1,064 ms, 225,248 B (65.1× smaller than uncompressed) |
| Export RELATIVE `.npy` | uncompressed 501 ms, 14,660,524 B; Draco 1,622 ms, 581,260 B |

**FPS: not measured.**

## Limitations

- The pixel is found where the ray meets the **drawn** surface. On a coarse
  LOD tile the drawn surface can differ from the full-resolution one, so near
  steep edges (building walls) the identified pixel can differ from what a
  full-resolution mesh would give; zooming in (finer tiles) removes this.
- Standing at the terrain centre in first person selects every tile at full
  resolution on the 512 × 512 synthetic raster (≈ 522k triangles).
- First-person collision is a height floor, not full collision: walls are
  climbed, not blocked.
- Browser verification so far is headless Chrome (SwiftShader). **FPS has not
  been measured.**
- The only real Phase 5 diagnostics validate `phase4b/synthetic_dsm.tif`, so
  overlays are available only for the secondary Phase 6 product
  `synthetic_dsm.tif` with the DSM surface shown. No nDSM or DTM error or
  confidence product exists.
- On the 512 × 512 raster the 300 px panel is ~0.59 CSS px per raster pixel,
  so a 2D point can resolve to a neighbouring pixel of the one aimed at. The
  readout always names the pixel actually linked.
- Tinting tiles by LOD level replaces the surface colours and hides the overlay.
- Dev-server-only debug handles (`window.__depthwizard`,
  `window.__depthwizardExport`) are used by the headless verification. They are
  absent from production builds.
- glTF export: the whole export (and the Draco encode) runs on the main
  thread. The page is unresponsive while it runs: measured 0.2–0.7 s
  uncompressed and 1.0–1.6 s with Draco for 512 × 512 rasters in headless
  Chrome (varies between runs). There is no Web Worker.
- None of the real rasters used so far contains nodata (Phase 6: 0 invalid
  pixels; the Phase 3 `.npy`: 0 non-finite values). Nodata handling in the
  terrain, measurement, cursor, overlays and export is verified on test
  fixtures only.
- The measurement panel shows no uncertainty field. The linked-cursor readout
  states "Uncertainty: not yet measured"; the data summary states that no
  per-pixel uncertainty product exists.
- For a RELATIVE raster the linked-cursor "3D marker (world x, y, z)" row is
  the scene position, so its y includes the display exaggeration. The raw value
  is shown in its own row.
- Draco precision is tied to the mesh's largest extent. For a pixel-space
  RELATIVE `.npy` (512 px wide) the step is ≈ 0.0078 units; the measured maximum
  height error was 0.0039 units on a 0.01–2.83 range. For exact values, export
  uncompressed.
- Uncompressed exports are large (≈ 14 MB for 512 × 512); larger rasters grow
  proportionally, and nothing is decimated.
- The Draco checkbox resets to ON whenever a new raster is loaded.
- The only real test data is the SYNTHETIC Phase 6 fixture (no nodata pixels)
  and the Phase 3 `.npy` outputs. The synthetic source image is grayscale, so
  RGB texturing is verified on a test fixture only.
