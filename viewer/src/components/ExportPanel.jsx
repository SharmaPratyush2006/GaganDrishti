import { useEffect, useState } from 'react';

import { loadDracoEncoder } from '../export/dracoEncoder.js';
import { exportTerrainGlb } from '../export/gltfExport.js';

const fmtBytes = (n) => (n >= 1048576 ? `${(n / 1048576).toFixed(2)} MB` : `${(n / 1024).toFixed(1)} kB`) + ` (${n.toLocaleString()} bytes)`;

function download(bytes, fileName) {
  const url = URL.createObjectURL(new Blob([bytes], { type: 'model/gltf-binary' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * "Export .glb" for the displayed raster (full resolution; see gltfExport.js).
 * Status: ready -> exporting -> exported | failed. Only measured numbers are shown.
 */
export default function ExportPanel({ raster, sourceImage }) {
  const [draco, setDraco] = useState({ status: 'checking' });
  const [useDraco, setUseDraco] = useState(true);
  const [state, setState] = useState({ status: 'ready' });
  // Sizes actually written for THIS raster, per variant, so a ratio is shown only when both were measured.
  const [sizes, setSizes] = useState({});

  useEffect(() => {
    let live = true;
    loadDracoEncoder().then((r) => live && setDraco(r.ok ? { status: 'available', encoder: r.encoder } : { status: 'unavailable', reason: r.reason }));
    return () => { live = false; };
  }, []);
  useEffect(() => {
    setState({ status: 'ready' });
    setSizes({});
  }, [raster, sourceImage]);

  const canExport = Boolean(raster) && raster.mode !== null && state.status !== 'exporting';
  const wantDraco = useDraco && draco.status === 'available';

  async function run() {
    if (!canExport) return;
    if (useDraco && draco.status !== 'available') {
      setState({ status: 'failed', error: `Draco compression is ${draco.status === 'checking' ? 'still loading' : `unavailable: ${draco.reason}`}; untick it to export uncompressed` });
      return;
    }
    setState({ status: 'exporting' });
    await new Promise((r) => setTimeout(r, 0)); // let "exporting" paint before the (synchronous) encode
    try {
      const result = await exportTerrainGlb({ raster, sourceImage, draco: wantDraco, dracoEncoder: wantDraco ? draco.encoder : null });
      download(result.bytes, result.fileName);
      setSizes((s) => ({ ...s, [wantDraco ? 'draco' : 'plain']: result.stats.byteLength }));
      setState({ status: 'exported', result });
      if (import.meta.env?.DEV) window.__depthwizardExport = result; // headless verification only
    } catch (err) {
      setState({ status: 'failed', error: err?.message ?? String(err) });
    }
  }

  const r = state.result;
  const dracoLabel = draco.status === 'available' ? 'Draco encoder loaded'
    : draco.status === 'checking' ? 'loading Draco encoder…' : `Draco unavailable: ${draco.reason}`;
  return (
    <div className="export-panel" data-testid="export-panel">
      <button onClick={run} disabled={!canExport} data-testid="export-button">Export .glb</button>{' '}
      <label className="tint">
        <input type="checkbox" checked={useDraco} onChange={(e) => setUseDraco(e.target.checked)} data-testid="export-draco" />
        {' '}Draco compression (KHR_draco_mesh_compression, lossy quantisation)
      </label>
      <span className={draco.status === 'unavailable' ? 'error' : 'muted'} data-testid="export-draco-status">{dracoLabel}</span>
      <div data-testid="export-status" data-status={state.status}>
        {state.status === 'ready' && <span className="muted">Export: ready — full-resolution mesh of the displayed {raster?.productType ?? 'raster'}{raster?.mode === 'RELATIVE' ? ' (RELATIVE: raw unitless values, display exaggeration not applied)' : ''}</span>}
        {state.status === 'exporting' && <span>Export: exporting…</span>}
        {state.status === 'failed' && <span className="error">Export: failed — {state.error}</span>}
        {state.status === 'exported' && (
          <dl className="mesh-debug">
            <div data-export="status"><dt>Export</dt><dd>exported successfully: {r.fileName}</dd></div>
            <div data-export="file size"><dt>file size</dt><dd>{fmtBytes(r.stats.byteLength)}</dd></div>
            <div data-export="mesh"><dt>mesh</dt><dd>{r.stats.vertices.toLocaleString()} vertices, {r.stats.triangles.toLocaleString()} triangles ({r.meta.nodata.trianglesOmitted.toLocaleString()} omitted for nodata)</dd></div>
            <div data-export="draco"><dt>Draco</dt><dd>{r.stats.draco.verified
              ? `verified in the written file (${r.inspection.draco.compressedPrimitives}/${r.inspection.primitives} primitive, KHR_draco_mesh_compression required)`
              : 'not compressed'}</dd></div>
            <div data-export="compression"><dt>size vs other variant</dt><dd>{sizes.draco && sizes.plain
              ? `Draco ${fmtBytes(sizes.draco)} vs uncompressed ${fmtBytes(sizes.plain)} → ${(sizes.plain / sizes.draco).toFixed(1)}× smaller`
              : 'not measured (export both variants of this raster to compare)'}</dd></div>
            <div data-export="heights"><dt>heights</dt><dd>{r.meta.heightValues}; vertical scale applied ×1{r.meta.displayedVerticalScale !== 1 ? ` (viewer displays ×${r.meta.displayedVerticalScale.toPrecision(4)}, not applied)` : ''}</dd></div>
            <div data-export="texture"><dt>texture</dt><dd>{r.meta.texture.included ? `${r.meta.texture.content} (${r.meta.texture.png} PNG)` : `none — ${r.meta.texture.reason}`}</dd></div>
            <div data-export="time"><dt>export time</dt><dd>{r.stats.exportMs.toFixed(0)} ms (mesh + texture + {r.stats.draco.verified ? 'Draco encode + ' : ''}GLB write + check, this run)</dd></div>
          </dl>
        )}
      </div>
    </div>
  );
}
