import { useEffect, useMemo, useRef, useState } from 'react';

import { createCameraRig } from '../3d/cameraRig.js';
import { heightmapFromRaster } from '../3d/heightmap.js';
import { createLinkedCursor } from '../3d/linkedCursor.js';
import { createMeasureTool } from '../3d/measure.js';
import { buildOverlay, compositeTexels, DEFAULT_OVERLAY_OPACITY, diagnosticEligibility } from '../3d/overlays.js';
import { createScene, initialCameraPose } from '../3d/scene.js';
import { TerrainLOD } from '../3d/terrain.js';
import { createDataTexture, textureForRaster } from '../3d/texture.js';
import { PRODUCT_MEANINGS } from '../data/metadata.js';
import ExportPanel from './ExportPanel.jsx';
import LinkedCursorPanel from './LinkedCursorPanel.jsx';
import MeasurementPanel from './MeasurementPanel.jsx';
import OverlayControls from './OverlayControls.jsx';

const MODE_LABEL = { orbit: 'Orbit', firstPerson: 'First Person' };
// Viewport corner label for the displayed surface.
const BADGE = { ndsm: 'nDSM', dsm: 'DSM', dtm: 'DTM', relative_height: 'RELATIVE HEIGHT' };
const HINTS = {
  orbit: 'left-drag rotate · right-drag pan · wheel zoom',
  firstPerson: 'W/S forward/back · A/D strafe · Q/E down/up · Shift faster · drag to look',
};

/**
 * Canvas showing one height raster as a quadtree-LOD mesh, draped with the
 * same-grid source image when there is one; orbit or first-person navigation;
 * click-to-measure against `measureRaster` (the nDSM, or the relative raster);
 * Phase 5 error / confidence overlays where they genuinely describe the
 * displayed surface (overlays.js); a 2D <-> 3D linked cursor (linkedCursor.js).
 *
 * `diagnostics` = { error, confidence, products, notAvailable } from loadDemo;
 * every field may be missing (e.g. a Phase 3 .npy has none).
 */
export default function Viewer3D({ raster, sourceImage, measureRaster, showTexture, tintByLevel, onStats, diagnostics = {} }) {
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const live = useRef({ terrain: null, rig: null, measure: null, cursor: null, texture: null, overlayTexture: null, pose: null });
  // Built overlays for the current raster, so toggling / cursor moves never rebuild them.
  const overlayCache = useRef({ raster: null, built: {} });
  const [error, setError] = useState(null);
  const [navMode, setNavMode] = useState('orbit');
  const [measuring, setMeasuring] = useState(false);
  const [measurement, setMeasurement] = useState(null);
  const [switchMs, setSwitchMs] = useState(null);
  const [overlayKind, setOverlayKind] = useState(null); // null | 'error' | 'confidence'
  const [opacity, setOpacity] = useState(DEFAULT_OVERLAY_OPACITY);
  const [overlay, setOverlay] = useState(null);
  const [overlayMs, setOverlayMs] = useState(null);
  const [linkedOpen, setLinkedOpen] = useState(false);
  const [linkAligned, setLinkAligned] = useState(true);
  const [location, setLocation] = useState(null);
  const cursorWanted = useRef(false);
  cursorWanted.current = linkedOpen && linkAligned;

  const { error: errorRaster, confidence: confidenceRaster, products, notAvailable } = diagnostics;
  const eligibility = useMemo(() => {
    if (!raster) return null;
    const ctx = { surfaceRaster: raster, products: products ?? null, notAvailable: notAvailable ?? {} };
    return {
      error: diagnosticEligibility('error', { ...ctx, diagnostic: errorRaster }),
      confidence: diagnosticEligibility('confidence', { ...ctx, diagnostic: confidenceRaster }),
    };
  }, [raster, errorRaster, confidenceRaster, products, notAvailable]);

  // One renderer for the component's lifetime.
  useEffect(() => {
    try {
      sceneRef.current = createScene(canvasRef.current);
    } catch (err) {
      setError(`WebGL could not be started: ${err.message}`);
      return undefined;
    }
    const observer = new ResizeObserver(() => sceneRef.current?.render());
    observer.observe(canvasRef.current);
    return () => {
      observer.disconnect();
      teardown(live.current);
      sceneRef.current?.dispose();
      sceneRef.current = null;
    };
  }, []);

  // Rebuild terrain, texture, navigation and the measure tool when the data changes.
  useEffect(() => {
    const view = sceneRef.current;
    if (!view || !raster) return;
    try {
      teardown(live.current);
      const t0 = performance.now();
      const heightmap = heightmapFromRaster(raster);
      const t1 = performance.now();
      const terrain = new TerrainLOD(heightmap);
      terrain.setTintByLevel(tintByLevel);
      const tex = textureForRaster(raster, sourceImage);
      terrain.setTexture(showTexture ? tex.texture : null);
      const t2 = performance.now();
      const pose = view.frame(terrain.bounds());
      view.setTerrain(terrain.group);
      const base = {
        heightmap,
        pose: initialCameraPose(terrain.bounds()),
        texture: { status: tex.status, label: tex.label, source: tex.source, alignment: tex.alignment,
          nodataTexels: tex.nodataTexels, buildMs: tex.buildMs },
        timings: { heightmapMs: t1 - t0, quadtreeAndTextureMs: t2 - t1 },
      };
      const rig = createCameraRig({
        camera: view.camera,
        domElement: canvasRef.current,
        terrain,
        render: () => view.render(),
        onUpdate: ({ lod, camera }) => onStats?.({ ...base, lod, camera }),
      });
      const measure = createMeasureTool({
        camera: view.camera, domElement: canvasRef.current, terrain, scene: view.scene,
        measureRaster, render: () => view.render(), onMeasure: setMeasurement,
      });
      const cursor = createLinkedCursor({
        camera: view.camera, domElement: canvasRef.current, terrain, scene: view.scene,
        render: () => view.render(), onLocate: setLocation,
      });
      cursor.setEnabled(cursorWanted.current);
      live.current = { terrain, rig, measure, cursor, texture: tex.texture, overlayTexture: null, pose };
      // Dev-server-only handle for headless browser verification; absent from production builds.
      if (import.meta.env?.DEV) window.__depthwizard = { live: live.current, view };
      const t3 = performance.now();
      rig.reset(pose); // first LOD selection + first render
      base.timings.firstLodAndRenderMs = performance.now() - t3;
      onStats?.({ ...base, lod: terrain.lastStats, camera: rig.status() });
      setNavMode('orbit');
      setMeasuring(false);
      setMeasurement(null);
      setLocation(null);
      setError(null);
    } catch (err) {
      setError(`Mesh could not be built: ${err.message}`);
      onStats?.(null);
    }
    // Tint and texture visibility are applied by the effects below without a rebuild.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raster, sourceImage, measureRaster]);

  useEffect(() => {
    const { terrain } = live.current;
    if (!terrain) return;
    terrain.setTintByLevel(tintByLevel);
    sceneRef.current?.render();
  }, [tintByLevel]);

  // Surface appearance = source texture and/or the active overlay. Only the
  // material changes: geometry, heights, LOD selection and tile cache are untouched.
  useEffect(() => {
    const l = live.current;
    if (!l.terrain || !raster) return;
    const t0 = performance.now();
    if (overlayCache.current.raster !== raster) overlayCache.current = { raster, built: {} };
    let ov = null;
    if (overlayKind && eligibility?.[overlayKind]?.available) {
      const cache = overlayCache.current.built;
      cache[overlayKind] ??= buildOverlay(overlayKind, {
        surfaceRaster: raster, diagnostic: diagnostics[overlayKind], products: products ?? null,
        notAvailable: notAvailable ?? {}, confidenceStates: products?.confidenceStates ?? null,
      });
      ov = cache[overlayKind].available ? cache[overlayKind] : null;
    }
    const previous = l.overlayTexture;
    if (ov) {
      const base = showTexture && l.texture ? l.texture.image.data : null;
      l.overlayTexture = createDataTexture({ width: raster.width, height: raster.height, data: compositeTexels(base, ov.overlay.data, opacity) });
      l.terrain.setTexture(l.overlayTexture);
    } else {
      l.overlayTexture = null;
      l.terrain.setTexture(showTexture ? l.texture : null);
    }
    previous?.dispose();
    sceneRef.current?.render();
    setOverlay(ov);
    if (overlayKind || previous) setOverlayMs(performance.now() - t0);
    // `diagnostics` fields are covered through `eligibility`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showTexture, overlayKind, opacity, raster, eligibility]);

  useEffect(() => {
    const { cursor } = live.current;
    if (!cursor) return;
    cursor.setEnabled(linkedOpen && linkAligned);
    if (!(linkedOpen && linkAligned)) setLocation(null);
  }, [linkedOpen, linkAligned]);

  const switchMode = (next) => {
    const { rig } = live.current;
    if (!rig) return;
    const t0 = performance.now();
    rig.setMode(next);
    setSwitchMs(performance.now() - t0);
    setNavMode(next);
    if (next === 'firstPerson') canvasRef.current?.focus();
  };
  const toggleMeasure = () => {
    const { measure } = live.current;
    if (!measure) return;
    measure.setActive(!measuring);
    setMeasuring(!measuring);
  };
  const toggleOverlay = (kind) => setOverlayKind((k) => (k === kind ? null : kind));
  const pickPixel = (row, col) => live.current.cursor?.setPixel(row, col);
  const clearCursor = () => live.current.cursor?.clear();
  const resetView = () => {
    const { rig, pose } = live.current;
    if (rig && pose) rig.reset(pose);
  };

  const measureLabel = measureRaster ? `${measureRaster.productType} — ${PRODUCT_MEANINGS[measureRaster.productType]}` : 'nothing (no nDSM or relative raster loaded)';

  const badge = raster ? (BADGE[raster.productType] ?? raster.productType) : null;

  return (
    <div className="viewer3d">
      <div className="viewport">
        <canvas ref={canvasRef} className="viewer-canvas" data-testid="terrain-canvas" tabIndex={0} />
        {badge && (
          <div className="viewport-badge" aria-hidden="true">
            <span className="badge-title">3D SURFACE</span>
            <span>{badge} · {raster.mode === 'RELATIVE' ? 'RELATIVE' : 'RECONSTRUCTED'}</span>
          </div>
        )}
        {measuring && <div className="viewport-mode" aria-hidden="true">MEASURE · click the terrain</div>}
      </div>
      <div className="subsection">
        <h3 className="section-label">3D Controls</h3>
        <div className="viewer-toolbar">
          <span className="camera-status" data-testid="camera-status">Camera: {MODE_LABEL[navMode]}</span>
          <div className="btn-group" role="group" aria-label="Camera mode">
            <button className={navMode === 'orbit' ? 'on' : ''} aria-pressed={navMode === 'orbit'} onClick={() => switchMode('orbit')}>Orbit</button>
            <button className={navMode === 'firstPerson' ? 'on' : ''} aria-pressed={navMode === 'firstPerson'} onClick={() => switchMode('firstPerson')}>First Person</button>
          </div>
          <button onClick={resetView}>Reset View</button>
          <button className={measuring ? 'on' : ''} aria-pressed={measuring} onClick={toggleMeasure} disabled={!measureRaster} data-testid="measure-toggle">
            {measuring ? 'Measure: ON (click to exit)' : 'Measure'}
          </button>
          <button className={linkedOpen ? 'on' : ''} aria-pressed={linkedOpen} onClick={() => setLinkedOpen(!linkedOpen)} data-testid="linked-toggle">
            {linkedOpen ? '2D Linked Cursor: ON' : '2D Linked Cursor'}
          </button>
        </div>
        <div className="muted hint">
          {HINTS[navMode]}{measuring ? ' · click the terrain to measure' : ''}
          {switchMs !== null && <> · last mode switch {switchMs.toFixed(1)} ms</>}
        </div>
        <MeasurementPanel active={measuring} result={measurement} measureLabel={measureLabel} />
      </div>
      {eligibility && (
        <div className="subsection">
          <h3 className="section-label">Overlays</h3>
          <OverlayControls eligibility={eligibility} active={overlayKind} onToggle={toggleOverlay} opacity={opacity}
            onOpacity={setOpacity} overlay={overlay} tintByLevel={tintByLevel} lastMs={overlayMs} />
        </div>
      )}
      {linkedOpen && raster && (
        <LinkedCursorPanel surfaceRaster={raster} sourceImage={sourceImage} overlay={overlay}
          overlayRaster={overlay ? diagnostics[overlay.kind] : null} opacity={opacity} location={location}
          onPick={pickPixel} onClear={clearCursor} onLinkState={setLinkAligned} />
      )}
      <div className="subsection">
        <h3 className="section-label">Export</h3>
        <ExportPanel raster={error ? null : raster} sourceImage={sourceImage} />
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

function teardown(l) {
  l.cursor?.dispose();
  l.overlayTexture?.dispose();
  l.cursor = null;
  l.overlayTexture = null;
  l.measure?.dispose();
  l.rig?.dispose();
  l.terrain?.dispose();
  l.texture?.dispose();
  l.measure = null;
  l.rig = null;
  l.terrain = null;
  l.texture = null;
}
