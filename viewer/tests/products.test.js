import { describe, expect, it } from 'vitest';

import { normalizeReportPath, reportPathToUrl, samePath } from '../src/data/paths.js';
import { PHASE5_SOURCE_LABEL, reportPhase, resolveDemoProducts } from '../src/data/products.js';

describe('report paths', () => {
  it('normalises Windows separators and maps to the data route', () => {
    expect(normalizeReportPath('data\\outputs\\phase6\\synthetic\\a\\ndsm.tif')).toBe('data/outputs/phase6/synthetic/a/ndsm.tif');
    expect(reportPathToUrl('.\\data\\outputs\\x y\\f.tif')).toBe('/data/outputs/x%20y/f.tif');
    expect(samePath('data\\outputs\\a.tif', 'data/outputs/a.tif')).toBe(true);
  });
  it('refuses absolute paths, escapes and paths outside data/outputs', () => {
    expect(() => normalizeReportPath('D:\\DepthWizard\\data\\outputs\\a.tif')).toThrow(/absolute/);
    expect(() => normalizeReportPath('/etc/passwd')).toThrow(/absolute/);
    expect(() => normalizeReportPath('data\\outputs\\..\\..\\secret')).toThrow(/escapes/);
    expect(() => reportPathToUrl('data\\raw\\x.tif')).toThrow(/not under data\/outputs/);
  });
  it('recognises report phases', () => {
    expect(reportPhase('data/outputs/phase4b/phase4b_report.json')).toBe('4b');
    expect(reportPhase('data\\outputs\\phase6\\synthetic\\phase6_report.json')).toBe('6');
    expect(reportPhase('data/outputs/phase6/other.json')).toBeNull();
  });
});

// Minimal reports with the same shape (and Windows paths) as the real ones.
const exports = (dir) => Object.fromEntries(['dsm.tif', 'dtm.tif', 'ndsm.tif', 'ground_mask.tif']
  .map((f) => [f, { path: `data\\outputs\\phase6\\s\\${dir}\\${f}`, ok: true }]));
const r6 = {
  path: 'data/outputs/phase6/s/phase6_report.json',
  json: {
    label: 'SYNTHETIC: specified, not measured',
    definitions: { DSM: 'absolute surface elevation' },
    acceptance: { input: 'b.tif' },
    reproducibility: { inputs: { phase4b_report: 'data\\outputs\\phase4b\\phase4b_report.json' } },
    products: {
      'a.tif': { role: 'secondary diagnostic', source_dsm: 'data\\outputs\\phase4b\\a.tif', exports: exports('a') },
      'b.tif': { role: 'ACCEPTANCE INPUT', source_dsm: 'data\\outputs\\phase4b\\b.tif', exports: exports('b') },
    },
  },
};
const r4b = { path: 'data/outputs/phase4b/phase4b_report.json', json: { image_grid: { source: 'data\\outputs\\phase4b\\fixture\\img.tif' } } };
const r5 = {
  path: 'data/outputs/phase5/s/phase5_report.json',
  json: {
    products: [
      { product: 'dsm', prediction_path: 'data\\outputs\\phase4b\\a.tif', error_map: { raster: { path: 'data\\outputs\\phase5\\s\\error_map_dsm.tif' } },
        confidence_raster: 'data\\outputs\\phase5\\s\\confidence_dsm.tif', confidence: { states: { INVALID: { code: 0, meaning: 'x' } } } },
      { product: 'agl', prediction_path: 'data\\outputs\\phase4b\\agl.tif', error_map: { raster: { path: 'data\\outputs\\phase5\\s\\error_map_agl.tif' } } },
    ],
  },
};

describe('resolveDemoProducts', () => {
  it('defaults to the Phase 6 acceptance input and resolves every surface from the report', () => {
    const p = resolveDemoProducts([r6, r4b, r5]);
    expect(p.productName).toBe('b.tif');
    expect(p.role).toBe('ACCEPTANCE INPUT');
    expect(p.surfaces.ndsm.url).toBe('/data/outputs/phase6/s/b/ndsm.tif');
    expect(p.surfaces.dsm.url).toBe('/data/outputs/phase6/s/b/dsm.tif');
    expect(p.surfaces.ndsm.url).not.toBe(p.surfaces.dsm.url);
    expect(p.image.url).toBe('/data/outputs/phase4b/fixture/img.tif');
    expect(p.synthetic).toBe(true);
  });

  it('reports Phase 5 layers as not available when Phase 5 never validated this source DSM', () => {
    const p = resolveDemoProducts([r6, r4b, r5]);
    expect(p.errorMap).toBeNull();
    expect(p.confidence).toBeNull();
    expect(p.notAvailable.error).toMatch(/no Phase 5 report validated/);
    expect(p.notAvailable.confidence).toMatch(/no Phase 5 report validated/);
    // The reason names what Phase 5 did validate, with normalised paths.
    expect(p.notAvailable.error).toContain('Phase 5 diagnostics exist only for data/outputs/phase4b/a.tif (Phase 6 product a.tif), not for this product');
    expect(p.notAvailable.error).not.toContain('\\');
  });

  it('links Phase 5 layers by prediction_path == source_dsm and labels them as source-DSM diagnostics', () => {
    const p = resolveDemoProducts([r6, r4b, r5], { productName: 'a.tif' });
    expect(p.errorMap.url).toBe('/data/outputs/phase5/s/error_map_dsm.tif');
    expect(p.errorMap.label).toBe(PHASE5_SOURCE_LABEL);
    expect(p.confidence.url).toBe('/data/outputs/phase5/s/confidence_dsm.tif');
    expect(p.confidenceStates.INVALID.code).toBe(0);
    expect(PHASE5_SOURCE_LABEL).toMatch(/NOT nDSM accuracy/);
  });

  it('never provides an uncertainty layer', () => {
    expect(resolveDemoProducts([r6, r4b, r5]).notAvailable.uncertainty).toMatch(/no per-pixel uncertainty/);
  });

  it('handles missing optional reports gracefully', () => {
    const p = resolveDemoProducts([r6]);
    expect(p.surfaces.ndsm).toBeTruthy();
    expect(p.image).toBeNull();
    expect(p.notAvailable.image).toMatch(/not under data\/outputs/);
    expect(p.errorMap).toBeNull();
  });

  it('refuses ambiguous Phase 5 links instead of picking one', () => {
    const dup = { ...r5, path: 'data/outputs/phase5/t/phase5_report.json' };
    const p = resolveDemoProducts([r6, r4b, r5, dup], { productName: 'a.tif' });
    expect(p.errorMap).toBeNull();
    expect(p.notAvailable.error).toMatch(/ambiguous/);
  });

  it('fails clearly without a Phase 6 report or with an unknown product', () => {
    expect(() => resolveDemoProducts([r4b])).toThrow(/no phase6_report.json/);
    expect(() => resolveDemoProducts([r6], { productName: 'zzz' })).toThrow(/not in a.tif, b.tif/);
  });
});
