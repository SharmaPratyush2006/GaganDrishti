import { useEffect, useMemo, useRef, useState } from 'react';

import { linkAlignment, northInPanel, panelLayout, panelToPixel, pixelToPanel } from '../3d/linkedCursor.js';
import { compositeTexels, overlayValueAt } from '../3d/overlays.js';
import { describeSourceImage, textureData } from '../3d/texture.js';
import { formatValue, PRODUCT_MEANINGS } from '../data/metadata.js';
import { validStats } from '../data/raster.js';

const PANEL_PX = 300;
const SHOWN = { ndsm: 'nDSM', dsm: 'DSM', dtm: 'DTM', relative_height: 'Phase 3 relative raster' };

/** RGBA texels of the displayed height raster on a linear grayscale display ramp (min..max). */
function surfaceTexels(r) {
  const { min, max } = validStats(r);
  const span = max !== null && max > min ? max - min : 1;
  const data = new Uint8Array(r.width * r.height * 4);
  for (let i = 0; i < r.width * r.height; i++) {
    if (!r.valid[i]) continue; // transparent: nodata is not drawn
    const g = Math.round(((Number(r.values[i]) - min) / span) * 255);
    data[i * 4] = g; data[i * 4 + 1] = g; data[i * 4 + 2] = g; data[i * 4 + 3] = 255;
  }
  return { data, min, max };
}

/**
 * Compact 2D raster view with a crosshair linked to the 3D marker.
 * The link is a raster (row, col) of the displayed surface (see linkedCursor.js).
 */
export default function LinkedCursorPanel({ surfaceRaster, sourceImage, overlay, overlayRaster, opacity, location, onPick, onClear, onLinkState }) {
  const canvasRef = useRef(null);
  const baseRef = useRef(null);
  const pendingRef = useRef(null);
  const [baseChoice, setBaseChoice] = useState('image');
  const [drawMs, setDrawMs] = useState(null);

  const img = describeSourceImage(sourceImage);
  const imgAlign = sourceImage ? linkAlignment(surfaceRaster, sourceImage) : null;
  const imageUsable = (img.kind === 'grayscale' || img.kind === 'rgb');
  const useImage = baseChoice === 'image' && imageUsable;
  const panelRaster = useImage ? sourceImage : surfaceRaster;
  const align = linkAlignment(surfaceRaster, panelRaster);
  const shown = SHOWN[surfaceRaster.productType] ?? surfaceRaster.productType;

  // The 3D side links only while this 2D product is on the displayed grid.
  useEffect(() => {
    onLinkState?.(align.aligned);
  }, [align.aligned, onLinkState]);

  // Base texels (+ active overlay, blended exactly as on the terrain).
  const texels = useMemo(() => {
    if (!align.aligned) return null;
    const base = useImage ? textureData(sourceImage, img.kind).data : surfaceTexels(surfaceRaster).data;
    return overlay?.available ? compositeTexels(base, overlay.overlay.data, opacity) : base;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [surfaceRaster, sourceImage, useImage, overlay, opacity, align.aligned]);

  const layout = panelLayout(surfaceRaster.width, surfaceRaster.height, PANEL_PX, PANEL_PX);
  const north = northInPanel(surfaceRaster.transform);

  // The raster at 1 texel per pixel, drawn once per content change.
  useEffect(() => {
    if (!texels) { baseRef.current = null; return; }
    const off = document.createElement('canvas');
    off.width = surfaceRaster.width;
    off.height = surfaceRaster.height;
    off.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(texels.buffer.slice(0)), surfaceRaster.width, surfaceRaster.height), 0, 0);
    baseRef.current = off;
  }, [texels, surfaceRaster]);

  // Redraw the view + crosshair.
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const t0 = performance.now();
    const dpr = window.devicePixelRatio || 1;
    cv.width = PANEL_PX * dpr;
    cv.height = PANEL_PX * dpr;
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0e1116';
    ctx.fillRect(0, 0, PANEL_PX, PANEL_PX);
    if (!baseRef.current) return;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(baseRef.current, layout.offX, layout.offY, surfaceRaster.width * layout.scale, surfaceRaster.height * layout.scale);
    if (north) {
      const [nx, ny] = north;
      const cx = PANEL_PX - 22;
      const cy = 24;
      ctx.strokeStyle = '#ffffff';
      ctx.fillStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - nx * 12, cy - ny * 12);
      ctx.lineTo(cx + nx * 12, cy + ny * 12);
      ctx.stroke();
      ctx.font = '11px system-ui';
      ctx.fillText('N', cx + nx * 16 - 4, cy + ny * 16 + 4);
    }
    if (location) {
      const [x, y] = pixelToPanel(layout, location.row, location.col);
      const cell = Math.max(layout.scale, 3);
      ctx.strokeStyle = '#39d5ff';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, layout.offY); ctx.lineTo(x, layout.offY + surfaceRaster.height * layout.scale);
      ctx.moveTo(layout.offX, y); ctx.lineTo(layout.offX + surfaceRaster.width * layout.scale, y);
      ctx.stroke();
      ctx.lineWidth = 2;
      ctx.strokeRect(x - cell / 2 - 2, y - cell / 2 - 2, cell + 4, cell + 4);
    }
    setDrawMs(performance.now() - t0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [texels, location, surfaceRaster]);

  const pixelFromEvent = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    return panelToPixel(layout, ((e.clientX - r.left) / r.width) * PANEL_PX, ((e.clientY - r.top) / r.height) * PANEL_PX);
  };
  const onMove = (e) => {
    if (!align.aligned) return;
    const px = pixelFromEvent(e);
    if (!px) return;
    // At most one 2D -> 3D update per animation frame.
    const first = pendingRef.current === null;
    pendingRef.current = px;
    if (first) {
      requestAnimationFrame(() => {
        const p = pendingRef.current;
        pendingRef.current = null;
        if (p) onPick(p.row, p.col);
      });
    }
  };
  const onClick = (e) => {
    if (!align.aligned) return;
    const px = pixelFromEvent(e);
    if (px) onPick(px.row, px.col);
  };

  const baseLabel = useImage
    ? (img.kind === 'grayscale' ? 'grayscale source image' : 'RGB source image')
    : `${shown} values — linear grayscale display ramp (nodata not drawn)`;
  const ov = overlay?.available ? overlayValueAt(overlay, overlayRaster, location?.row ?? -1, location?.col ?? -1) : null;
  const rows = location ? [
    ['raster pixel', `row ${location.row}, col ${location.col}`],
    ['map coordinate (pixel centre)', location.map ? `E ${location.map[0].toFixed(2)}, N ${location.map[1].toFixed(2)} (${location.crs ?? 'CRS not declared'})` : 'not georeferenced (pixel space)'],
    ['3D marker (world x, y, z)', location.world.map((v) => (v === null ? 'nodata' : v.toFixed(2))).join(', ')],
    [`${shown} value`, location.value === null ? 'nodata — no valid value at this pixel' : formatValue(location.value, surfaceRaster, 2)],
    ...(ov ? [[overlay.label, ov.text]] : []),
    ['Uncertainty', 'not yet measured'],
    ['last update', `${location.from === '2d' ? '2D → 3D' : location.from === '3d-hover' ? '3D hover → 2D' : '3D click → 2D'}${location.responseMs != null ? `, ${location.responseMs.toFixed(1)} ms` : ''}`],
  ] : [];

  return (
    <div className="linked-panel" data-testid="linked-panel">
      <div className="linked-head">
        <strong>2D linked cursor</strong>
        <label>
          2D view:{' '}
          <select value={useImage ? 'image' : 'surface'} onChange={(e) => setBaseChoice(e.target.value)} data-testid="linked-base">
            <option value="image" disabled={!imageUsable}>source image{imageUsable ? ` (${img.kind === 'grayscale' ? 'grayscale' : 'RGB'})` : ' (unavailable)'}</option>
            <option value="surface">{shown} values</option>
          </select>
        </label>
        <button onClick={onClear} disabled={!location}>Clear cursor</button>
      </div>
      <div className="muted" data-testid="linked-base-label">
        Showing: {baseLabel}{overlay?.available ? ` + ${overlay.label}` : ''}. Raster grid as stored (row 0 at top){north ? '; arrow = map north from the transform' : ''}.
      </div>
      {imgAlign && !imgAlign.aligned && <div className="muted">Source image not linkable: {imgAlign.reasons.join('; ')}</div>}
      {align.aligned ? (
        <canvas ref={canvasRef} className="linked-canvas" data-testid="linked-canvas" style={{ width: PANEL_PX, height: PANEL_PX }}
          onPointerMove={onMove} onClick={onClick} />
      ) : (
        <p className="error" data-testid="linked-disabled">Linked cursor disabled — the 2D product is not aligned with the 3D surface ({align.reasons.join('; ')}).</p>
      )}
      <div data-testid="linked-readout">
        {location ? (
          <dl className="mesh-debug">
            {rows.map(([k, v]) => (<div key={k} data-linked={k}><dt>{k}</dt><dd>{v}</dd></div>))}
          </dl>
        ) : <p className="muted">Move over the 2D view or the 3D terrain (click also works).</p>}
        <div className="muted" title={PRODUCT_MEANINGS[surfaceRaster.productType]}>
          linked grid: {surfaceRaster.width} × {surfaceRaster.height} ({shown}){drawMs !== null ? ` · 2D redraw ${drawMs.toFixed(1)} ms` : ''}
        </div>
      </div>
    </div>
  );
}
