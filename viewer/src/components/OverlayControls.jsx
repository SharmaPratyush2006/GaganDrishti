import { divergingColor } from '../3d/overlays.js';

const rgb = (c) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
const signed = (v, units) => `${v > 0 ? '+' : ''}${v.toFixed(3)}${units === 'm' ? ' m' : ''}`;

/** Legend of the active overlay. Error: diverging ramp with its real limits; confidence: the documented states. */
function Legend({ ov }) {
  if (ov.kind === 'error') {
    const o = ov.overlay;
    const stops = [-1, -0.5, 0, 0.5, 1].map((t) => rgb(divergingColor(t, 1)));
    return (
      <div className="legend" data-testid="error-legend">
        <div className="legend-title">{ov.label}</div>
        <div className="ramp" style={{ background: `linear-gradient(to right, ${stops.join(', ')})` }} />
        <div className="ramp-labels"><span>−{o.limit.toFixed(3)}{ov.units === 'm' ? ' m' : ''}</span><span>0</span><span>+{o.limit.toFixed(3)}{ov.units === 'm' ? ' m' : ''}</span></div>
        <div className="muted">{ov.meaning}. Colour saturates at ±{o.limit.toFixed(3)} (99th percentile of |error|; {o.counts.saturated.toLocaleString()} px beyond). Recorded range {signed(o.min, ov.units)} … {signed(o.max, ov.units)}.</div>
        <div className="muted">Pixels: {o.counts.positive.toLocaleString()} positive, {o.counts.negative.toLocaleString()} negative, {o.counts.zero.toLocaleString()} zero, {o.counts.invalid.toLocaleString()} nodata/invalid (not drawn).</div>
        <div className="muted">Source: {ov.source}</div>
      </div>
    );
  }
  const o = ov.overlay;
  return (
    <div className="legend" data-testid="confidence-legend">
      <div className="legend-title">{ov.label}</div>
      <div className="disclaimer" data-testid="confidence-disclaimer">{ov.disclaimer}</div>
      <table>
        <thead><tr><th /><th>code</th><th>state</th><th>meaning (Phase 5)</th><th>pixels</th></tr></thead>
        <tbody>
          {ov.states.map((s) => (
            <tr key={s.code} data-state={s.name}>
              <td><span className="swatch" style={{ background: rgb(s.color) }} /></td>
              <td>{s.code}</td><td>{s.name}</td><td>{s.meaning}</td><td>{o.counts[s.name].toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="muted">{o.nodata.toLocaleString()} nodata px, {o.undocumented.toLocaleString()} px with undocumented codes (not drawn). Source: {ov.source}</div>
    </div>
  );
}

/** Error / Confidence overlay toggles with their availability, plus the legend of the active one. */
export default function OverlayControls({ eligibility, active, onToggle, opacity, onOpacity, overlay, tintByLevel, lastMs }) {
  const row = (kind, title) => {
    const el = eligibility[kind];
    const on = active === kind;
    return (
      <div className="overlay-row" data-testid={`${kind}-overlay`}>
        <button className={on ? 'on' : ''} onClick={() => onToggle(kind)} data-testid={`${kind}-toggle`}>
          {title}: {on ? 'ON' : 'OFF'}
        </button>{' '}
        <span className={el.available ? 'ok' : 'muted'} data-testid={`${kind}-status`}>
          {el.available ? `available — ${kind === 'error' ? 'Phase 5 DSM error — prediction minus reference' : 'Phase 5 DSM rule-based quality flags'} (aligned with the displayed DSM)` : el.reason}
        </span>
      </div>
    );
  };
  return (
    <div className="overlays" data-testid="overlays">
      {row('error', 'Error overlay')}
      {row('confidence', 'Confidence overlay')}
      <div className="muted">One overlay at a time; the overlay only colours the surface — terrain heights are unchanged.</div>
      {overlay?.available && (
        <>
          <label className="tint">
            overlay opacity{' '}
            <input type="range" min="0.1" max="1" step="0.05" value={opacity} onChange={(e) => onOpacity(Number(e.target.value))} />
            {' '}{Math.round(opacity * 100)}%
          </label>
          {tintByLevel && <span className="error"> LOD tinting is on: it replaces the surface colours, so the overlay is hidden.</span>}
          <Legend ov={overlay} />
        </>
      )}
      {lastMs !== null && <div className="muted" data-testid="overlay-ms">last overlay change {lastMs.toFixed(1)} ms (build + blend + material swap + render)</div>}
    </div>
  );
}
