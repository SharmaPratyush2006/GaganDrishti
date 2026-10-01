import { Fragment, useEffect, useMemo, useState } from 'react';

import DataSummary from './components/DataSummary.jsx';
import DemoPanel from './components/DemoPanel.jsx';
import MeshDebugPanel from './components/MeshDebugPanel.jsx';
import Viewer3D from './components/Viewer3D.jsx';
import { chooseNpy, DEFAULT_PHASE3_NPY, readDemoManifest, startupLoad } from './data/demo.js';
import { fetchIndex, fetchNpy, loadDemo } from './data/load.js';
import { PRODUCT_MEANINGS } from './data/metadata.js';
import { reportPhase } from './data/products.js';

const SURFACES = ['ndsm', 'dsm', 'dtm'];

// User-facing pipeline stages. Internally: 1 = Phase 3 relative height, 2 = Phase 6 DSM/DTM, 3 = nDSM in 3D.
const STAGES = [
  { n: 1, name: 'Terrain & Height Estimation', desc: 'Estimate relative surface height from a single satellite image.' },
  { n: 2, name: 'Ground Separation', desc: 'Separate ground elevation from above-ground structures.' },
  { n: 3, name: 'Building Height & 3D Analysis', desc: 'Visualize the reconstructed surface and measure building heights in 3D.' },
];

// Explicit selection (never index order): ?phase6Report=<path>&product=<name>, ?npy=<path>, ?demo=1.
// The bare root URL loads the Phase 8 demo (startupLoad in demo.js).
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

  // No .npy chosen: select and load the named Phase 3 default (never the first in the index).
  const loadRelative = (chosen = npyPath) => run(async () => {
    if (!chosen) {
      chosen = DEFAULT_PHASE3_NPY;
      setNpyPath(chosen);
    }
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
    const startup = startupLoad(PARAMS);
    if (startup === 'demo') loadPhase8Demo();
    else if (startup === 'phase6') loadPhase6(PARAMS.get('product') ?? undefined, PARAMS.get('phase6Report'));
    else if (startup === 'npy') loadRelative(PARAMS.get('npy'));
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

  const busy = state.status === 'loading';
  // Display only: which pipeline stage the loaded data belongs to (internal phases unchanged).
  const activeStage = !loaded ? null : state.relative ? 1 : surface === 'ndsm' ? 3 : 2;
  const status = busy ? { kind: 'busy', text: 'LOADING' }
    : state.status === 'error' ? { kind: 'warn', text: 'LOAD FAILED' }
      : loaded && state.demo ? { kind: 'ok', text: 'DEMO MODE' } : { kind: 'ok', text: 'SYSTEM READY' };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>DEPTHWIZARD</h1>
          <span className="subtitle">Single-View Satellite → 3D Reconstruction</span>
        </div>
        <span className={`status-pill ${status.kind}`} role="status"><span className="dot" aria-hidden="true">●</span> {status.text}</span>
      </header>

      <nav className="pipeline" aria-label="Processing pipeline">
        <div className="stage input">
          <span className="stage-num">IN</span>
          <span className="stage-name">Satellite Image</span>
        </div>
        {STAGES.map((st) => (
          <Fragment key={st.n}>
            <span className="pipe-arrow" aria-hidden="true">→</span>
            <div className={`stage${activeStage === st.n ? ' active' : ''}`} aria-current={activeStage === st.n ? 'step' : undefined}>
              <span className="stage-num">0{st.n}{activeStage === st.n && <span className="stage-tag">ACTIVE</span>}</span>
              <span className="stage-name">{st.name}</span>
            </div>
          </Fragment>
        ))}
      </nav>

      <div className="workspace">
        <aside className="sidebar">
          <section className="card actions">
            <h2 className="card-title">Data Source</h2>
            <button className="primary" onClick={loadPhase8Demo} disabled={busy}>▶ Load 3D Building Demo</button>
            <p className="card-desc">Loads the recorded demo report and its surfaces.</p>
          </section>

          <section className="card">
            <h2 className="card-title"><span className="card-num">01</span>{STAGES[0].name}</h2>
            <p className="card-desc">{STAGES[0].desc}</p>
            {index && index.npy.length > 1 && (
              <label className="field">
                <span className="field-label">Relative .npy</span>
                <select value={npyPath} onChange={(e) => setNpyPath(e.target.value)} data-testid="npy-choice" title={npyPath || undefined}>
                  <option value="">— choose —</option>
                  {index.npy.map((p) => <option key={p} value={p} title={p}>{shortPath(p)}</option>)}
                </select>
              </label>
            )}
            <button onClick={() => loadRelative()} disabled={busy}>Load Terrain &amp; Height</button>
          </section>

          <section className="card">
            <h2 className="card-title"><span className="card-num">02</span>{STAGES[1].name}</h2>
            <p className="card-desc">{STAGES[1].desc}</p>
            {phase6Reports.length > 1 && (
              <label className="field">
                <span className="field-label">Report</span>
                <select value={phase6Report} onChange={(e) => setPhase6Report(e.target.value)} data-testid="phase6-report" title={phase6Report || undefined}>
                  <option value="">— choose —</option>
                  {phase6Reports.map((p) => <option key={p} value={p} title={p}>{shortPath(p)}</option>)}
                </select>
              </label>
            )}
            <button onClick={() => loadPhase6(productName)} disabled={busy}>Load Ground Separation</button>
            {state.products && (
              <label className="field">
                <span className="field-label">Product</span>
                <select value={state.products.productName} onChange={(e) => loadPhase6(e.target.value, state.products.phase6Report)}>
                  {state.products.availableProducts.map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </label>
            )}
            {loaded && state.rasters && (
              <fieldset className="surface-choice">
                <legend className="field-label">Surface</legend>
                <div className="segmented">
                  {SURFACES.map((s) => (
                    <label key={s} title={PRODUCT_MEANINGS[s]}
                      className={`seg${surface === s ? ' on' : ''}${state.rasters[s] ? '' : ' disabled'}`}>
                      <input type="radio" name="surface" value={s} checked={surface === s} disabled={!state.rasters[s]}
                        onChange={() => setSurface(s)} />
                      {s === 'ndsm' ? 'nDSM' : s.toUpperCase()}
                    </label>
                  ))}
                </div>
                <p className="card-desc">{PRODUCT_MEANINGS[surface]}</p>
              </fieldset>
            )}
          </section>
        </aside>

        <main className="main">
          <section className="card stage-card">
            <h2 className="card-title"><span className="card-num">03</span>{STAGES[2].name}</h2>
            <p className="card-desc">{STAGES[2].desc}</p>
            {busy && <p className="notice">Loading…</p>}
            {state.status === 'error' && <p className="error notice">Load failed: {state.message}</p>}
            {meshRaster ? (
              <div className="mesh-section">
                <Viewer3D raster={meshRaster} sourceImage={state.rasters?.image ?? null} measureRaster={measureRaster} showTexture={showTexture}
                  tintByLevel={tint} onStats={setMeshInfo} diagnostics={diagnostics} />
                <div className="subsection">
                  <h3 className="section-label">Display</h3>
                  <label className="tint">
                    <input type="checkbox" checked={showTexture} onChange={(e) => setShowTexture(e.target.checked)} /> show source image texture
                  </label>
                  <label className="tint">
                    <input type="checkbox" checked={tint} onChange={(e) => setTint(e.target.checked)} /> tint tiles by LOD level (replaces the texture)
                  </label>
                  {meshInfo && (
                    <details className="diagnostics">
                      <summary>Mesh &amp; LOD diagnostics</summary>
                      <MeshDebugPanel info={meshInfo} />
                    </details>
                  )}
                </div>
              </div>
            ) : !busy && (
              <div className="viewport-empty">No surface loaded — choose a data source on the left.</div>
            )}
          </section>
          {loaded && state.demo && <DemoPanel manifest={state.demo.manifest} path={state.demo.path} />}
          {loaded && <section className="card summary-card"><DataSummary state={state} /></section>}
        </main>
      </div>
    </div>
  );
}

/** Repo path without the data/outputs/ prefix, for compact dropdown text (the value stays the full path). */
const shortPath = (p) => p.replace(/^data\/outputs\//, '');
