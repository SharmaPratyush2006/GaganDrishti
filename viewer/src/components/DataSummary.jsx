import { PRODUCT_MEANINGS, unitSuffix } from '../data/metadata.js';
import { validStats } from '../data/raster.js';

function RasterRow({ type, raster, aligned, label }) {
  const s = validStats(raster);
  const unit = unitSuffix(raster);
  const range = s.min === null ? 'no valid pixel' : `${s.min.toFixed(3)} … ${s.max.toFixed(3)}${unit}`;
  return (
    <tr>
      <td>{type}</td>
      <td>
        {PRODUCT_MEANINGS[raster.productType]}
        {label && <div className="layer-label">{label}</div>}
      </td>
      <td>{raster.width} × {raster.height}</td>
      <td>{raster.dtype}</td>
      <td>{raster.nodata === null ? 'none declared' : String(raster.nodata)}</td>
      <td>{s.validPixels.toLocaleString()} / {s.nodataPixels.toLocaleString()}</td>
      <td>{range}</td>
      <td>{raster.crs ?? 'none'}</td>
      <td>{raster.transform ? `[${raster.transform.join(', ')}]` : 'none'}</td>
      <td>{raster.units ?? (raster.mode === 'RELATIVE' ? 'unitless' : 'not declared')}</td>
      <td title={raster.modeReason}>{raster.mode ?? 'n/a'}</td>
      <td>{aligned === undefined ? '—' : aligned ? 'yes' : 'NO'}</td>
      <td>{raster.warnings.join('; ')}</td>
    </tr>
  );
}

export default function DataSummary({ state }) {
  const rows = state.relative ? { relative_height: state.relative } : state.rasters;
  return (
    <section>
      {state.products && (
        <div className="meta">
          {state.products.synthetic && <p className="synthetic">{state.products.label}</p>}
          <p>
            Phase 6 report <code>{state.products.phase6Report}</code>, product <b>{state.products.productName}</b> ({state.products.role});
            source DSM <code>{state.products.sourceDsm}</code>
          </p>
        </div>
      )}
      <table>
        <thead>
          <tr>
            <th>layer</th><th>meaning</th><th>size</th><th>dtype</th><th>nodata</th><th>valid / nodata px</th>
            <th>valid range</th><th>CRS</th><th>transform</th><th>units</th><th>mode</th><th>on nDSM grid</th><th>warnings</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(rows).map(([type, r]) => (
            <RasterRow key={type} type={type} raster={r} aligned={state.alignment?.[type]}
              label={{ error: state.products?.errorMap?.label, confidence: state.products?.confidence?.label }[type]} />
          ))}
        </tbody>
      </table>
      {state.notAvailable && (
        <ul className="missing">
          {Object.entries(state.notAvailable).map(([k, v]) => (
            <li key={k}><b>{k}</b>: not available — {v}</li>
          ))}
        </ul>
      )}
      {state.relative && <p>Source: <code>{state.relative.source}</code>. RELATIVE: values are unitless, never metres.</p>}
      <p className="timing">Load + decode time (this browser, this run): {state.loadMs.toFixed(0)} ms</p>
    </section>
  );
}
