import { useMemo, useState } from 'react';

import DataSummary from './components/DataSummary.jsx';
import MeshDebugPanel from './components/MeshDebugPanel.jsx';
import Viewer3D from './components/Viewer3D.jsx';
import { fetchIndex, fetchNpy, loadDemo } from './data/load.js';
import { PRODUCT_MEANINGS } from './data/metadata.js';

const SURFACES = ['ndsm', 'dsm', 'dtm'];

export default function App() {
  const [state, setState] = useState({ status: 'idle' });
  const [productName, setProductName] = useState(undefined);
  const [surface, setSurface] = useState('ndsm');
  const [tint, setTint] = useState(false);
  const [showTexture, setShowTexture] = useState(true);
  const [meshInfo, setMeshInfo] = useState(null);

  async function run(load) {
    setState({ status: 'loading' });
    setMeshInfo(null);
    try {
      setState({ status: 'loaded', ...(await load()) });
    } catch (err) {
      setState({ status: 'error', message: err.message });
    }
  }

  const loadPhase6 = (name) => {
    setProductName(name);
    run(() => loadDemo({ productName: name }));
  };

  const loadRelative = () => run(async () => {
    const t0 = performance.now();
    const index = await fetchIndex();
    if (index.npy.length === 0) throw new Error('no .npy relative-height output found under data/outputs');
    const raster = await fetchNpy(index.npy[0]);
    return { relative: raster, loadMs: performance.now() - t0 };
  });

  const loaded = state.status === 'loaded';
  const meshRaster = loaded ? (state.relative ?? state.rasters?.[surface] ?? null) : null;
  // Click-to-measure reads the nDSM whatever surface is shown; a .npy has no nDSM, so its own relative values.
  const measureRaster = loaded ? (state.rasters?.ndsm ?? state.relative ?? null) : null;
  // Phase 5 layers + the links that say what they validated (products.js); none for a .npy.
  const diagnostics = useMemo(() => ({
    error: state.rasters?.error ?? null,
    confidence: state.rasters?.confidence ?? null,
    products: state.products ?? null,
    notAvailable: state.notAvailable ?? (state.relative ? { error: 'a Phase 3 relative raster has no Phase 5 error map', confidence: 'a Phase 3 relative raster has no Phase 5 confidence raster' } : {}),
  }), [state]);

  return (
    <div className="app">
      <header>
        <h1>DepthWizard 3D Viewer</h1>
        <span className="step">Step 6: quadtree-LOD mesh · source-image texture · orbit + first-person · click-to-measure (nDSM) · Phase 5 error/confidence overlays · 2D↔3D linked cursor</span>
      </header>
      <div className="actions">
        <button onClick={() => loadPhase6(productName)} disabled={state.status === 'loading'}>Load Phase 6 demo</button>
        {state.products && (
          <label>
            Phase 6 product:{' '}
            <select value={state.products.productName} onChange={(e) => loadPhase6(e.target.value)}>
              {state.products.availableProducts.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </label>
        )}
        <button onClick={loadRelative} disabled={state.status === 'loading'}>Load a Phase 3 relative .npy</button>
      </div>
      {loaded && state.rasters && (
        <fieldset className="surface-choice">
          <legend>Surface to mesh</legend>
          {SURFACES.map((s) => (
            <label key={s} title={PRODUCT_MEANINGS[s]}>
              <input type="radio" name="surface" value={s} checked={surface === s} disabled={!state.rasters[s]}
                onChange={() => setSurface(s)} />
              {' '}{s === 'ndsm' ? 'nDSM' : s.toUpperCase()} <span className="muted">— {PRODUCT_MEANINGS[s]}</span>
            </label>
          ))}
        </fieldset>
      )}
      {state.status === 'loading' && <p>Loading…</p>}
      {state.status === 'error' && <p className="error">Load failed: {state.message}</p>}
      {meshRaster && (
        <section className="mesh-section">
          <Viewer3D raster={meshRaster} sourceImage={state.rasters?.image ?? null} measureRaster={measureRaster} showTexture={showTexture}
            tintByLevel={tint} onStats={setMeshInfo} diagnostics={diagnostics} />
          <label className="tint">
            <input type="checkbox" checked={showTexture} onChange={(e) => setShowTexture(e.target.checked)} /> show source image texture
          </label>{' '}
          <label className="tint">
            <input type="checkbox" checked={tint} onChange={(e) => setTint(e.target.checked)} /> tint tiles by LOD level (replaces the texture)
          </label>
          <MeshDebugPanel info={meshInfo} />
        </section>
      )}
      {loaded && <DataSummary state={state} />}
    </div>
  );
}
