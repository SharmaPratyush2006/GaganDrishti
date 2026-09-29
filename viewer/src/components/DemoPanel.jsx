/**
 * Phase 8 demo manifest (demo/reports/demo_report.json), shown as recorded.
 *
 * In plain words: every number here was written by the Python pipeline
 * (`python -m depthwizard.demo synthetic`). Nothing is recomputed in the
 * browser: the shadow-physics rows are the Phase 2 pipeline's own shadow
 * lengths and heights, compared with the fixture's SPECIFIED heights.
 */
import { reportPathToUrl } from '../data/paths.js';

const NOT_MEASURED = 'not yet measured';

/** Recorded number -> text; a missing value is "not yet measured", never 0. */
function fmt(v, digits = 3) {
  if (v === null || v === undefined) return NOT_MEASURED;
  return typeof v === 'number' ? v.toFixed(digits) : String(v);
}

export default function DemoPanel({ manifest, path }) {
  const md = manifest.input.metadata;
  const p2 = manifest.phase2_shadow_physics;
  const clicks = manifest.phase6_surfaces.clicks;
  const figureUrl = p2.figure ? reportPathToUrl(p2.figure) : null;
  // The DSM error-map figure Phase 5 wrote (first product that has one).
  const withFigure = manifest.phase5_validation.products.find((p) => p.product === 'dsm' && p.error_map_figure)
    ?? manifest.phase5_validation.products.find((p) => p.error_map_figure);
  const errorFigure = withFigure ? { url: reportPathToUrl(withFigure.error_map_figure), product: withFigure.product.toUpperCase() } : null;
  return (
    <section className="demo-panel" data-testid="demo-panel">
      <h2>Phase 8 demo <span className="muted">— <code>{path}</code></span></h2>
      <p className="synthetic">{manifest.label}</p>

      <div data-testid="demo-input">
      <h3>Input (Phase 1)</h3>
      <dl className="mesh-debug">
        <div><dt>mode</dt><dd><b>{manifest.mode}</b> (heights in {manifest.units})</dd></div>
        <div><dt>file</dt><dd><code>{manifest.input.image}</code> — {md.width_px} × {md.height_px} px, {md.band_count} band(s), {md.dtype}</dd></div>
        <div><dt>CRS</dt><dd>{md.crs ?? 'none'}</dd></div>
        <div><dt>GSD</dt><dd>{md.gsd_m === null ? 'none' : `${fmt(md.gsd_m)} m`}</dd></div>
        <div><dt>sun elevation</dt><dd>{md.sun_elevation_deg === null ? 'none' : `${fmt(md.sun_elevation_deg)}°`} <span className="muted">← {md.sun_elevation_source ?? '—'}</span></dd></div>
        <div><dt>sun azimuth</dt><dd>{md.sun_azimuth_deg === null ? 'none' : `${fmt(md.sun_azimuth_deg)}°`} <span className="muted">← {md.sun_azimuth_source ?? '—'}</span></dd></div>
        <div><dt>routing</dt><dd>{manifest.input.routing.reason}</dd></div>
      </dl>
      </div>

      <div data-testid="demo-shadow">
      <h3>Shadow Physics — Synthetic Fixture (Phase 2)</h3>
      <p className="muted">
        h = L · tan(θ), θ = sun elevation. Footprints: {p2.footprints}. Reference heights are the fixture&apos;s
        SPECIFIED heights (synthetic ground truth), not real-world measurements.
      </p>
      <table>
        <thead>
          <tr>
            <th>building</th><th>shadow L (px)</th><th>shadow L (m)</th><th>tan θ</th><th>h = L·tan θ (m)</th>
            <th>specified (m)</th><th>|error| (m)</th><th>status</th>
          </tr>
        </thead>
        <tbody>
          {p2.buildings.map((b) => (
            <tr key={b.building}>
              <td>{b.building}</td><td>{fmt(b.shadow_length_px, 2)}</td><td>{fmt(b.shadow_length_m)}</td>
              <td>{fmt(b.tan_elevation, 4)}</td><td>{fmt(b.height_m)}</td><td>{fmt(b.specified_height_m, 1)}</td>
              <td>{fmt(b.abs_error_m)}</td><td>{b.confidence}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        SYNTHETIC MAE {fmt(p2.mae_m)} m, max {fmt(p2.max_abs_error_m)} m ({p2.succeeded} of {p2.buildings.length} buildings);
        pixel-grid bound {fmt(p2.pixel_grid_bound_m)} m — {p2.error_source}.
      </p>
      {figureUrl && <img className="demo-figure" src={figureUrl} alt="Phase 2 shadow measurement on the synthetic fixture" />}
      </div>

      <h3>nDSM at the building centroids (Phase 6, synthetic)</h3>
      <table>
        <thead><tr><th>building</th><th>specified H (m)</th><th>nDSM (m)</th><th>|error| (m)</th></tr></thead>
        <tbody>
          {clicks.map((c) => (
            <tr key={c.building}><td>{c.building}</td><td>{fmt(c.specified_height_m, 1)}</td><td>{fmt(c.ndsm_m, 4)}</td><td>{fmt(c.abs_error_m, 6)}</td></tr>
          ))}
        </tbody>
      </table>
      <p className="muted">Click a building in the 3D view: the measurement panel reads the same nDSM raster.</p>

      <div data-testid="demo-validation">
      <h3>Validation (Phase 5, synthetic)</h3>
      <ul>
        {manifest.phase5_validation.products.map((p) => {
          // The unit goes on recorded numbers only, never on "not yet measured" or a reason.
          const m = (v) => (typeof v === 'number' && p.units === 'metres' ? `${fmt(v)} m` : fmt(v));
          return (
            <li key={p.product}>
              {p.product.toUpperCase()} vs specified truth: overall MAE {m(p.overall_mae)}, RMSE {m(p.overall_rmse)}, building MAE {m(p.building_mae)}
            </li>
          );
        })}
        <li>Real-world validation: {manifest.phase5_validation.real_world_validation}</li>
        <li>Overlays: {manifest.viewer.overlays}</li>
      </ul>
      {errorFigure && <img className="demo-figure" src={errorFigure.url} alt={`Phase 5 ${errorFigure.product} error map (synthetic)`} />}
      </div>

      <h3>Not yet measured</h3>
      <ul className="missing">
        <li>Uncertainty: {manifest.uncertainty}</li>
        {Object.entries(manifest.not_yet_measured).map(([k, v]) => <li key={k}>{k.replace(/_/g, ' ')}: {v}</li>)}
      </ul>
    </section>
  );
}
