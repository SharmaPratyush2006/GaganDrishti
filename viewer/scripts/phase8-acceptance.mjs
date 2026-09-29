#!/usr/bin/env node
/**
 * Phase 8 browser acceptance + screenshots, over the Chrome DevTools Protocol.
 *
 * In plain words: opens the running viewer in headless Chrome, loads the Phase 8
 * demo, checks what a judge would see (metadata, shadow-physics table, 3D
 * scene, click-to-measure = raster value, overlays, glTF/Draco export,
 * RELATIVE .npy stays unitless), and saves cropped screenshots of the real
 * application. No screenshot is edited or composed.
 *
 * Needs: the demo outputs (`python -m depthwizard.demo synthetic`) and the Vite
 * dev server (`npm run dev` in viewer/). Optionally the API (`--api`).
 *
 *   node scripts/phase8-acceptance.mjs [--base http://localhost:5173] [--api http://127.0.0.1:8000]
 *        [--out ../data/outputs/demo/screenshots] [--chrome <path>] [--npy <repo-relative .npy>]
 *
 * Writes <out>/*.png and <out>/acceptance.json (git-ignored). Exit 1 if any check fails.
 * Uses the dev-only debug handles (window.__depthwizard, __depthwizardExport),
 * so it runs against `vite` (dev), not a production build. FPS is not measured.
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const here = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(here, '..', '..');
const { values: args } = parseArgs({
  options: {
    base: { type: 'string', default: 'http://localhost:5173' },
    api: { type: 'string' },
    out: { type: 'string', default: path.join(REPO, 'data', 'outputs', 'demo', 'screenshots') },
    chrome: { type: 'string', default: process.env.CHROME_PATH },
    npy: { type: 'string', default: 'data/outputs/phase3/verification/JAX_004_006_r256_c256_relative_height.npy' },
    port: { type: 'string', default: '9333' },
  },
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function findChrome() {
  const candidates = [
    args.chrome,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
  ].filter(Boolean);
  const found = candidates.find((p) => fs.existsSync(p));
  if (!found) throw new Error('Chrome not found; pass --chrome <path> or set CHROME_PATH');
  return found;
}

const results = [];
function check(name, ok, detail = '') {
  results.push({ name, ok: Boolean(ok), detail: String(detail) });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  — ${detail}` : ''}`);
}

// ---------------------------------------------------------------- CDP plumbing
let ws;
let nextId = 0;
const pending = new Map();
const listeners = [];
async function connect(port) {
  for (let i = 0; i < 100 && !ws; i++) {
    try {
      const page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page');
      if (page) ws = new WebSocket(page.webSocketDebuggerUrl);
    } catch { /* Chrome still starting */ }
    if (!ws) await sleep(200);
  }
  if (!ws) throw new Error('could not connect to Chrome');
  await new Promise((r) => ws.addEventListener('open', r));
  ws.addEventListener('message', (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    } else if (msg.method) listeners.forEach((f) => f(msg));
  });
}
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++nextId;
  pending.set(id, (msg) => (msg.error ? reject(new Error(`${method}: ${msg.error.message}`)) : resolve(msg.result)));
  ws.send(JSON.stringify({ id, method, params }));
});
async function evaluate(expression) {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(`evaluate: ${r.exceptionDetails.exception?.description ?? r.exceptionDetails.text}`);
  return r.result.value;
}
async function waitFor(expression, timeout = 30000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    if (await evaluate(expression)) return true;
    await sleep(100);
  }
  return false;
}
async function goto(url) {
  await send('Page.navigate', { url });
  await waitFor("document.readyState === 'complete' && !!document.querySelector('h1')");
}
const text = (sel) => evaluate(`document.querySelector(${JSON.stringify(sel)})?.innerText ?? null`);
const clickButton = (label) => evaluate(`(() => {
  const b = [...document.querySelectorAll('button')].find((x) => x.textContent.trim() === ${JSON.stringify(label)});
  if (!b || b.disabled) return false; b.click(); return true; })()`);
const clickTestId = (id) => evaluate(`(() => { const e = document.querySelector('[data-testid=${id}]'); if (!e) return false; e.click(); return true; })()`);
/** Set the <select> that offers `value` (React-visible change event). */
async function selectValue(value) {
  return evaluate(`(() => {
    const s = [...document.querySelectorAll('select')].find((x) => [...x.options].some((o) => o.value === ${JSON.stringify(value)}));
    if (!s) return false;
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, ${JSON.stringify(value)});
    s.dispatchEvent(new Event('change', { bubbles: true })); return true; })()`);
}
async function mouseClick(x, y) {
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 });
}

/** Screenshot of the union of the selectors' boxes (page coordinates), padded. */
async function shot(name, selectors, pad = 8) {
  const rect = await evaluate(`(() => {
    const rs = ${JSON.stringify(selectors)}.flatMap((s) => [...document.querySelectorAll(s)]).map((e) => e.getBoundingClientRect()).filter((r) => r.width && r.height);
    if (!rs.length) return null;
    const x0 = Math.min(...rs.map((r) => r.left)), y0 = Math.min(...rs.map((r) => r.top));
    const x1 = Math.max(...rs.map((r) => r.right)), y1 = Math.max(...rs.map((r) => r.bottom));
    return { x: x0 + scrollX, y: y0 + scrollY, width: x1 - x0, height: y1 - y0 }; })()`);
  if (!rect) return check(`screenshot ${name}`, false, `nothing matched ${selectors.join(', ')}`);
  const clip = { x: Math.max(0, rect.x - pad), y: Math.max(0, rect.y - pad), width: rect.width + 2 * pad, height: rect.height + 2 * pad, scale: 1 };
  const { data } = await send('Page.captureScreenshot', { format: 'png', clip, captureBeyondViewport: true });
  const file = path.join(args.out, `${name}.png`);
  fs.writeFileSync(file, Buffer.from(data, 'base64'));
  check(`screenshot ${name}`, true, path.relative(REPO, file).split(path.sep).join('/'));
}

/** Screen position of a map coordinate at height h on the displayed terrain (dev-only handle). */
const projectExpr = (e, n, h) => `(() => {
  const d = window.__depthwizard; const hm = d.live.terrain.heightmap; const cam = d.view.camera;
  const v = cam.position.clone().set(${e} - hm.origin[0], ${h} * hm.verticalScale, -(${n} - hm.origin[1])).project(cam);
  const r = document.querySelector('[data-testid=terrain-canvas]').getBoundingClientRect();
  return { x: r.left + (v.x + 1) / 2 * r.width, y: r.top + (1 - v.y) / 2 * r.height, inView: Math.abs(v.x) < 1 && Math.abs(v.y) < 1 };
})()`;
async function measureAt(e, n, h) {
  const p = await evaluate(projectExpr(e, n, h));
  await evaluate("window.__lastMeasure = document.querySelector('[data-testid=measurement]')?.innerText ?? ''");
  await mouseClick(p.x, p.y);
  await waitFor("(document.querySelector('[data-testid=measurement]')?.innerText ?? '') !== window.__lastMeasure"
    + " && /raw raster value/.test(document.querySelector('[data-testid=measurement]')?.innerText ?? '')", 10000);
  const rows = Object.fromEntries(await evaluate(
    "[...document.querySelectorAll('[data-testid=measurement] [data-measure]')].map((d) => [d.dataset.measure, d.querySelector('dd').innerText])"));
  return { p, rows, title: await text('[data-testid=measurement] .measurement-title') };
}
const summaryRows = () => evaluate("[...document.querySelectorAll('table tbody tr')].filter((tr) => tr.cells.length >= 11).map((tr) => ({ layer: tr.cells[0].innerText, units: tr.cells[9].innerText, mode: tr.cells[10].innerText }))");

// ---------------------------------------------------------------- run
fs.mkdirSync(args.out, { recursive: true });
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'dw-phase8-chrome-'));
const chrome = spawn(findChrome(), ['--headless=new', `--remote-debugging-port=${args.port}`, `--user-data-dir=${profile}`,
  '--window-size=1600,2400', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--no-first-run',
  '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' });

const exceptions = [];
const consoleErrors = [];
const httpErrors = [];
try {
  if (args.api) {
    try {
      const h = await (await fetch(`${args.api}/health`)).json();
      check('API /health', h.status === 'ok' && h.project === 'DepthWizard' && h.phase === 8, JSON.stringify(h));
    } catch (err) {
      check('API /health', false, err.message);
    }
  }
  const manifestPath = path.join(REPO, 'data', 'outputs', 'demo', 'reports', 'demo_report.json');
  if (!fs.existsSync(manifestPath)) throw new Error(`${manifestPath} missing: run \`python -m depthwizard.demo synthetic\` first`);
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

  await connect(args.port);
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Network.enable');
  const urls = new Map();
  listeners.push((m) => {
    if (m.method === 'Runtime.exceptionThrown') exceptions.push(m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text);
    if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') consoleErrors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
    if (m.method === 'Network.requestWillBeSent') urls.set(m.params.requestId, m.params.request.url);
    if (m.method === 'Network.responseReceived' && m.params.response.status >= 400) httpErrors.push(`${m.params.response.status} ${m.params.response.url}`);
    if (m.method === 'Network.loadingFailed' && !m.params.canceled) httpErrors.push(`FAILED ${m.params.errorText} ${urls.get(m.params.requestId) ?? ''}`);
  });

  // 1. viewer + Phase 8 demo
  await goto(`${args.base}/`);
  check('viewer loads', (await text('h1')) === 'DepthWizard 3D Viewer');
  check('"Load Phase 8 Demo" present', await waitFor("[...document.querySelectorAll('button')].some((b) => b.textContent === 'Load Phase 8 Demo')", 10000));
  await clickButton('Load Phase 8 Demo');
  check('Phase 8 demo loads', await waitFor("!!document.querySelector('[data-testid=demo-panel]') && !!window.__depthwizard?.live?.terrain", 60000));
  await waitFor("[...document.querySelectorAll('.demo-figure')].every((i) => i.complete && i.naturalWidth > 0)", 10000);

  const md = manifest.input.metadata;
  const input = (await text('[data-testid=demo-input]')) ?? '';
  check('input metadata: ABSOLUTE, CRS, GSD, sun (from the manifest)', input.includes('ABSOLUTE') && input.includes(md.crs)
    && input.includes(`${md.gsd_m.toFixed(3)} m`) && input.includes(`${md.sun_elevation_deg.toFixed(3)}°`) && input.includes(`${md.sun_azimuth_deg.toFixed(3)}°`),
  `${md.crs}, GSD ${md.gsd_m} m, sun ${md.sun_elevation_deg}° / ${md.sun_azimuth_deg}°`);
  const shadow = (await text('[data-testid=demo-shadow]')) ?? '';
  const p2 = manifest.phase2_shadow_physics;
  check('shadow physics labelled synthetic', shadow.includes('Shadow Physics — Synthetic Fixture') && shadow.includes('synthetic ground truth'));
  check('shadow table = manifest (L px, h, specified, error)', p2.buildings.every((b) => shadow.includes(b.shadow_length_px.toFixed(2))
    && shadow.includes(b.height_m.toFixed(3)) && shadow.includes(b.abs_error_m.toFixed(3))),
  p2.buildings.map((b) => `${b.building} ${b.height_m.toFixed(3)} m`).join(', '));
  check('uncertainty: not yet measured', ((await text('[data-testid=demo-panel]')) ?? '').includes('Uncertainty: not yet measured'));
  const rows = await summaryRows();
  check('DSM/DTM/nDSM: ABSOLUTE, metres', ['dsm', 'dtm', 'ndsm'].every((l) => rows.some((r) => r.layer === l && r.mode === 'ABSOLUTE' && /metre/.test(r.units))), JSON.stringify(rows.filter((r) => r.mode)));
  const tiles = await evaluate('window.__depthwizard.live.terrain.group.children.length');
  check('3D scene renders (LOD tiles)', tiles > 0, `${tiles} tiles`);

  await shot('01_input_metadata', ['.actions', '[data-testid=demo-input]']);
  await shot('02_shadow_physics', ['[data-testid=demo-shadow]']);
  await shot('04_terrain_3d', ['.surface-choice', '[data-testid=terrain-canvas]', '.viewer-toolbar']);

  // 2. click-to-measure at the centre of the pixel Phase 6 height_at() resolves
  const r6 = JSON.parse(fs.readFileSync(path.join(REPO, manifest.viewer.phase6_report), 'utf-8'));
  const [a, , c0, , e, f0] = r6.products[manifest.phase6_surfaces.acceptance_input].grid.transform;
  await clickTestId('measure-toggle');
  let last = null;
  for (const c of [...manifest.phase6_surfaces.clicks].sort((x, y) => x.ndsm_m - y.ndsm_m)) {
    const col = Math.floor((c.easting_m - c0) / a);
    const row = Math.floor((c.northing_m - f0) / e);
    const m = await measureAt(c0 + (col + 0.5) * a, f0 + (row + 0.5) * e, c.ndsm_m);
    const raw = Number((m.rows['raw raster value'] ?? '').split(' ')[0]);
    check(`click ${c.building}: nDSM = raster value (height_at)`, m.p.inView && m.rows['raster pixel'] === `row ${row}, col ${col}`
      && raw === c.ndsm_m && m.rows.units === 'metres', `${m.title}; raw ${raw}; ${m.rows['raster pixel']}`);
    last = c;
  }
  await shot('05_click_measure', ['[data-testid=terrain-canvas]', '.viewer-toolbar', '[data-testid=measurement]']);
  await clickTestId('measure-toggle');
  check('tallest building measured last (screenshot shows it)', last?.building === 'slab_c', last?.building);

  // 3. navigation + linked cursor
  await clickButton('First person');
  check('first-person mode', await waitFor("/First/i.test(document.querySelector('[data-testid=camera-status]').innerText)", 5000));
  await clickButton('Orbit');
  check('orbit mode', await waitFor("/Orbit/i.test(document.querySelector('[data-testid=camera-status]').innerText)", 5000));
  await clickTestId('linked-toggle');
  check('linked cursor opens', await waitFor("!!document.querySelector('[data-testid=linked-canvas]')", 5000));
  await clickTestId('linked-toggle');

  // 4. glTF / Draco export of the acceptance nDSM
  await waitFor("/loaded/.test(document.querySelector('[data-testid=export-draco-status]')?.innerText ?? '')", 30000);
  await evaluate("window.__depthwizardExport = null; document.querySelector('[data-testid=export-button]').click()");
  const exported = await waitFor('!!window.__depthwizardExport', 60000);
  const ex = exported ? await evaluate('({ verified: window.__depthwizardExport.stats.draco.verified, bytes: window.__depthwizardExport.stats.byteLength, mode: window.__depthwizardExport.meta.mode, units: window.__depthwizardExport.meta.units })') : null;
  check('glTF export with Draco verified', ex?.verified === true && ex.mode === 'ABSOLUTE' && ex.units === 'metres', JSON.stringify(ex));
  await waitFor("document.querySelector('[data-testid=export-status]')?.dataset.status === 'exported'", 10000);
  await shot('06_gltf_export', ['[data-testid=export-panel]']);

  // 5. validation: overlays keep their rule; the validated product shows the error map on its DSM
  check('overlays refused on the acceptance nDSM', /unavailable|not available/i.test((await text('[data-testid=error-status]')) ?? ''));
  await selectValue('synthetic_dsm.tif');
  await waitFor("[...document.querySelectorAll('select')].some((s) => s.value === 'synthetic_dsm.tif') && !!window.__depthwizard?.live?.terrain && !document.querySelector('[data-testid=demo-panel]')", 30000);
  await evaluate("document.querySelector('input[name=surface][value=dsm]').click()");
  await waitFor("/^available/.test(document.querySelector('[data-testid=error-status]')?.innerText ?? '')", 15000);
  check('synthetic_dsm.tif + DSM: error overlay available', /^available/.test((await text('[data-testid=error-status]')) ?? ''), await text('[data-testid=error-status]'));
  await clickTestId('error-toggle');
  check('error overlay legend', await waitFor("!!document.querySelector('[data-testid=error-legend]')", 10000));
  await shot('03_validation_error_map', ['[data-testid=terrain-canvas]', '[data-testid=overlays]']);
  await clickTestId('error-toggle');
  await clickTestId('confidence-toggle');
  check('confidence overlay legend + disclaimer', await waitFor("!!document.querySelector('[data-testid=confidence-legend]') && !!document.querySelector('[data-testid=confidence-disclaimer]')", 10000));
  await clickTestId('confidence-toggle');

  // 6. the demo's validation section (metrics + Phase 5 error map figure), as a second view
  await goto(`${args.base}/?demo=1`);
  check('?demo=1 loads the demo', await waitFor("!!document.querySelector('[data-testid=demo-validation]') && !!window.__depthwizard?.live?.terrain", 60000));
  await waitFor("[...document.querySelectorAll('.demo-figure')].every((i) => i.complete && i.naturalWidth > 0)", 10000);
  const val = (await text('[data-testid=demo-validation]')) ?? '';
  check('validation metrics from the manifest, labelled synthetic', val.includes('Validation (Phase 5, synthetic)') && val.includes('Real-world validation: not yet measured'));
  await shot('03b_validation_metrics', ['[data-testid=demo-validation]']);

  // 7. existing Phase 7 data, explicitly selected; no-choice refusal kept
  await goto(`${args.base}/?phase6Report=data/outputs/phase6/synthetic/phase6_report.json`);
  const p7 = await waitFor("!!window.__depthwizard?.live?.terrain && /phase6\\/synthetic\\/phase6_report/.test(document.body.innerText)", 60000);
  check('Phase 7 data via ?phase6Report (skipped if absent)', p7 || !fs.existsSync(path.join(REPO, 'data/outputs/phase6/synthetic/phase6_report.json')));
  await goto(`${args.base}/`);
  if (await waitFor("!!document.querySelector('[data-testid=phase6-report]')", 5000)) {
    await clickButton('Load Phase 6 demo');
    check('several Phase 6 reports + no choice -> explicit refusal', await waitFor("/several Phase 6 reports found; choose one/.test(document.querySelector('.error')?.innerText ?? '')", 10000));
  }

  // 8. RELATIVE .npy stays unitless
  if (fs.existsSync(path.join(REPO, args.npy))) {
    await goto(`${args.base}/?npy=${args.npy}`);
    check('RELATIVE .npy loads', await waitFor('!!window.__depthwizard?.live?.terrain', 60000));
    const rel = (await summaryRows())[0];
    check('RELATIVE: mode RELATIVE, unitless', rel?.mode === 'RELATIVE' && rel?.units === 'unitless', JSON.stringify(rel));
    await clickTestId('measure-toggle');
    const hm = await evaluate('window.__depthwizard.live.terrain.heightmap.origin');
    const m = await measureAt(hm[0], hm[1], 0);
    const all = `${m.title} ${Object.values(m.rows).join(' ')}`;
    check('RELATIVE measurement is never metres', /Relative height/.test(m.title ?? '') && !/\d m\b|metre/.test(all), m.title);
  } else {
    check('RELATIVE .npy (skipped: not present)', true, args.npy);
  }
} catch (err) {
  check('acceptance script completed', false, err.stack);
} finally {
  const unexpected = httpErrors.filter((h) => !/\/favicon\.ico$/.test(h));
  check('no JavaScript exceptions', exceptions.length === 0, exceptions.join(' || '));
  check('no console errors', consoleErrors.length === 0, consoleErrors.join(' || '));
  check('no failed data requests (favicon excepted)', unexpected.length === 0, `all HTTP errors: ${httpErrors.join(' || ') || 'none'}`);
  const passed = results.filter((r) => r.ok).length;
  const summary = { when: new Date().toISOString(), base: args.base, browser: 'headless Chrome (SwiftShader WebGL)', fps: 'not measured', passed, total: results.length, results };
  fs.writeFileSync(path.join(args.out, 'acceptance.json'), JSON.stringify(summary, null, 2));
  console.log(`\n${passed} / ${results.length} checks passed; screenshots and acceptance.json in ${args.out}`);
  try { ws?.close(); } catch { /* closing anyway */ }
  chrome.kill();
  try { fs.rmSync(profile, { recursive: true, force: true }); } catch { /* Chrome may still hold files */ }
  process.exit(passed === results.length ? 0 : 1);
}
