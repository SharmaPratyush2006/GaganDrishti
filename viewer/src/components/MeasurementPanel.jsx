/** The latest click-to-measure result. Values come from the raster; nothing is estimated. */
export default function MeasurementPanel({ active, result, measureLabel }) {
  if (!active) return null;
  if (!result) {
    return (
      <div className="measurement" data-testid="measurement">
        <div className="measurement-title">Measure: click the terrain</div>
        <div className="muted">Measuring: {measureLabel}</div>
      </div>
    );
  }
  const heightLabel = result.productType === 'ndsm' ? 'Height above ground' : result.productType === 'relative_height' ? 'Relative height' : 'Value';
  const rows = result.ok
    ? [
        [heightLabel, result.formatted],
        ['raw raster value', `${result.value} (float32 as stored)`],
      ]
    : [['result', `${result.message} — ${result.reason}`]];
  rows.push(['product', result.product ? `${result.productType} — ${result.product}` : 'n/a']);
  if (result.row !== undefined) rows.push(['raster pixel', `row ${result.row}, col ${result.col}`]);
  if (result.map) rows.push(['map coordinate', `E ${result.map[0].toFixed(2)}, N ${result.map[1].toFixed(2)} (${result.crs})`]);
  if (result.mode) rows.push(['mode', result.mode]);
  if (result.units) rows.push(['units', result.units]);
  if (result.displayed) rows.push([`displayed surface (${result.displayed.productType})`, `${result.displayed.formatted} — ${result.displayed.product}`]);
  if (result.tile) rows.push(['hit LOD tile', `${result.tile.nodeId} (stride ${result.tile.stride})`]);
  rows.push(['response', `${result.responseMs.toFixed(1)} ms (raycast + lookup, this click)`]);
  return (
    <div className={`measurement ${result.ok ? '' : 'invalid'}`} data-testid="measurement">
      <div className="measurement-title">{result.ok ? `${heightLabel}: ${result.formatted}` : result.message}</div>
      <dl className="mesh-debug">
        {rows.map(([k, v]) => (
          <div key={k} data-measure={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
