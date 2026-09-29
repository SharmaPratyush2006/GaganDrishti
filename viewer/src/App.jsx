import { useEffect, useMemo, useState } from 'react';

import DataSummary from './components/DataSummary.jsx';
import DemoPanel from './components/DemoPanel.jsx';
import MeshDebugPanel from './components/MeshDebugPanel.jsx';
import Viewer3D from './components/Viewer3D.jsx';
import { chooseNpy, readDemoManifest } from './data/demo.js';
import { fetchIndex, fetchNpy, loadDemo } from './data/load.js';
import { PRODUCT_MEANINGS } from './data/metadata.js';
import { reportPhase } from './data/products.js';

const SURFACES = ['ndsm', 'dsm', 'dtm'];

// Explicit selection (never index order): ?phase6Report=<path>&product=<name>, ?npy=<path>, ?demo=1.
const PARAMS = new URLSearchParams(window.location.search);

export default function App() {
  const [state, setState] = useState({ status: 'idle' });
  const [productName, setProductName] = useState(PARAMS.get('product') ?? undefined);
  const [index, setIndex] = useState(null);
  const [phase6Report, setPhase6Report] = useState(PARAMS.get('phase6Report') ?? '');
  const [npyPath, setNpyPath] = useState(PARAMS.get('npy') ?? '');
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

  // With several Phase 6 reports and none chosen, the loader refuses to guess (products.js).
  const loadPhase6 = (name, report = phase6Report) => {
    setProductName(name);
    run(() => loadDemo({ phase6Report: report || undefined, productName: name }));
  };

  const loadRelative = (chosen = npyPath) => run(async () => {
    const t0 = performance.now();
    const raster = await fetchNpy(chooseNpy(await fetchIndex(), chosen));
    return { relative: raster, loadMs: performance.now() - t0 };
  });

  // Phase 8: the demo manifest names its own Phase 6 report; load exactly that one.
  const loadPhase8Demo = () => run(async () => {
    const { manifest, path, phase6Report: report, productName: name } = await readDemoManifest();
    setPhase6Report(report);
    setProductName(name);
    const loaded = await loadDemo({ phase6Report: report, productName: name });
    return { ...loaded, demo: { manifest, path } };
  });

  useEffect(() => {
    fetchIndex().then((idx) => {
      setIndex(idx);
      const p6 = idx.reports.filter((p) => reportPhase(p) === '6');
      if (p6.length === 1) setPhase6Report((cur) => cur || p6[0]);
    }).catch(() => setIndex(null));
    if (PARAMS.get('demo') === '1') loadPhase8Demo();
    else if (PARAMS.get('phase6Report')) loadPhase6(PARAMS.get('product') ?? undefined, PARAMS.get('phase6Report'));
    else if (PARAMS.get('npy')) loadRelative(PARAMS.get('npy'));
    // Runs once: the URL is read at start-up only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const phase6Reports = index ? index.reports.filter((p) => reportPhase(p) === '6') : [];

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
        <button className="demo-button" onClick={loadPhase8Demo} disabled={state.status === 'loading'}>Load Phase 8 Demo</button>
        {phase6Reports.length > 1 && (
          <label>
            Phase 6 report:{' '}
            <select value={phase6Report} onChange={(e) => setPhase6Report(e.target.value)} data-testid="phase6-report">
              <option value="">— choose —</option>
              {phase6Reports.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
        )}
        <button onClick={() => loadPhase6(productName)} disabled={state.status === 'loading'}>Load Phase 6 demo</button>
        {state.products && (
          <label>
            Phase 6 product:{' '}
            <select value={state.products.productName} onChange={(e) => loadPhase6(e.target.value, state.products.phase6Report)}>
              {state.products.availableProducts.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </label>
        )}
        {index && index.npy.length > 1 && (
          <label>
            Relative .npy:{' '}
            <select value={npyPath} onChange={(e) => setNpyPath(e.target.value)} data-testid="npy-choice">
              <option value="">— choose —</option>
              {index.npy.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
        )}
        <button onClick={() => loadRelative()} disabled={state.status === 'loading'}>Load a Phase 3 relative .npy</button>
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
      {loaded && state.demo && <DemoPanel manifest={state.demo.manifest} path={state.demo.path} />}
      {loaded && <DataSummary state={state} />}
    </div>
  );
}
