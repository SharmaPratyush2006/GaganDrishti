# DepthWizard — 60-second judge demo

Every shot below is a screen that exists in the current application. Every
number is one the application displays, taken from
`data/outputs/demo/reports/demo_report.json` (written by
`python -m depthwizard.demo synthetic`). **All heights in this demo are
SYNTHETIC**: they were specified when the fixture was generated, not measured
in the real world.

## Before recording

```bash
python -m depthwizard.demo synthetic         # writes data/outputs/demo/
python -m depthwizard.demo serve             # API on http://127.0.0.1:8000
cd viewer && npm run dev                     # viewer on http://localhost:5173
```

Open two browser tabs:

1. `http://127.0.0.1:8000/docs`, with `POST /process` expanded and "Try it out" pressed.
2. `http://localhost:5173/?demo=1`

For stills of the same screens, run `npm run acceptance:phase8` in `viewer/`.
It writes cropped screenshots of the running app to
`data/outputs/demo/screenshots/` (git-ignored). The file names are listed per
shot below.

## Shot list

| Time | Screen | Action | Voice-over |
| --- | --- | --- | --- |
| **0–8 s** | Viewer tab, top of the page: "DepthWizard 3D Viewer" and the synthetic fixture texture | None | "Measuring building height usually needs stereo, lidar or radar. We start from one optical image, and the shadows already in it." |
| **8–20 s** | API tab `/docs` → `POST /process` | Upload `data/outputs/demo/input/synthetic_city_a.tif` and press Execute. Scroll the response: `"mode": "ABSOLUTE"`, `"metric_capable_input": true`, `crs` EPSG:32643, `gsd_m` 0.5, sun elevation 45° and azimuth 135°, each with its source tag | "We upload a GeoTIFF. The system reads the CRS, the 0.5 m ground sample distance, and the sun at 45° elevation and 135° azimuth from the file's own tags. So this image can support metric heights." |
|  | Cut to the viewer tab, **Input (Phase 1)** block of the demo panel (`01_input_metadata.png`) | None | "The same metadata drives the demo." |
| **20–35 s** | Demo panel, **Shadow Physics — Synthetic Fixture (Phase 2)** table and figure (`02_shadow_physics.png`) | Point at the `slab_c` row, then the MAE line | "Each shadow is measured along the anti-sun direction, and height is L times tan θ. slab_c: 90.51 px of shadow gives 45.255 m, against a specified 45 m. Across four buildings the error is 0.230 m on average, 0.343 m at most, inside the 0.354 m pixel-grid limit." |
| **35–50 s** | 3D view (`04_terrain_3d.png`) | Press **First person**, fly with `W`/`A`/`S`/`D` over the buildings, press **Orbit**. Press **Measure** and click the top of the tallest building (`05_click_measure.png`) | "This is the nDSM in 3D: height above the extracted ground. Click the tallest building and the viewer reads the raster itself: 45.00 m, specified 45 m, in metres, ABSOLUTE." |
| **50–60 s** | Demo panel, **Validation (Phase 5, synthetic)** block with the DSM error-map figure (`03b_validation_metrics.png`) | Point at the DSM line, then "Real-world validation: not yet measured" | "Every product is scored. On the synthetic fixture the DSM's overall error is 0.314 m, 0.162 m on buildings. Real-world validation of this chain is not yet measured, and we show that, not a number." |

Optional alternative for 50–60 s: select `synthetic_dsm.tif`, choose the
**DSM** surface and turn on the **Error overlay** (`03_validation_error_map.png`).
The overlay appears only on the surface it validates. On the nDSM it is
refused, with the reason on screen.

## Numbers used, and where they come from

| Shown | Value | Source field in `demo_report.json` |
| --- | --- | --- |
| CRS, GSD | EPSG:32643, 0.5 m | `input.metadata.crs`, `gsd_m` |
| Sun | 45° elevation, 135° azimuth | `input.metadata.sun_elevation_deg`, `sun_azimuth_deg` |
| slab_c shadow → height | 90.51 px → 45.255 m (specified 45 m) | `phase2_shadow_physics.buildings[]` |
| Phase 2 error (SYNTHETIC) | MAE 0.230 m, max 0.343 m, bound 0.354 m | `phase2_shadow_physics.mae_m`, `max_abs_error_m`, `pixel_grid_bound_m` |
| Click on slab_c (nDSM) | 45.0000 m (specified 45 m) | `phase6_surfaces.clicks[]` |
| DSM validation (SYNTHETIC) | overall MAE 0.314 m, RMSE 0.316 m, building MAE 0.162 m | `phase5_validation.products[]` (`dsm`) |

The click-to-measure value was checked against this report by the Phase 8
browser acceptance script (`05_click_measure`): the panel's raw raster value
equals `phase6_surfaces.clicks[].ndsm_m` at the same pixel.

## NOT YET MEASURED (say it, never replace it with a number)

- Real-world accuracy of this end-to-end chain
- Real-world DTM / nDSM / georeferenced DSM accuracy
- Per-pixel height uncertainty
- Viewer frame rate (FPS) and any rendering-speed claim

## What not to say

- Not "upload any GeoTIFF and get a metric DSM". The `/process` response in
  the 8–20 s shot lists shadow heights, DSM, DTM and nDSM as `not available`,
  because they need supplied building footprints. The metric chain shown here
  runs on the synthetic fixture.
- Not "the model predicted these heights". The DSM in this demo comes from a
  relative field constructed from the fixture's truth, not from Phase 3.
- Not "zero-shot" or "unseen city" for the Phase 5 DFC2019 result. That run
  was city-held-out with LiDAR-anchored per-tile calibration, and it is not
  part of this 60-second demo.
- No building detection: the footprints are supplied.
- No relative (Phase 3) value is ever given in metres.
