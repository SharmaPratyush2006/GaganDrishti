/**
 * Loads the Draco glTF encoder (draco3dgltf, Google's official WASM build)
 * in the browser, once, on first use.
 *
 * In plain words: the package's Emscripten build runs in both Node and
 * browsers; in a browser it cannot find its .wasm by itself, so the bytes are
 * fetched from the URL Vite serves for the package file and handed over as
 * `wasmBinary`. Any failure is reported as "Draco unavailable: <reason>" --
 * the caller then refuses a Draco export instead of silently writing an
 * uncompressed one.
 */
import encoderWasmUrl from 'draco3dgltf/draco_encoder.wasm?url';

let pending = null;

/** @returns {Promise<{ok:true, encoder:object}|{ok:false, reason:string}>} */
export function loadDracoEncoder() {
  pending ??= (async () => {
    try {
      if (typeof WebAssembly !== 'object') throw new Error('WebAssembly is not supported by this browser');
      const res = await fetch(encoderWasmUrl);
      if (!res.ok) throw new Error(`GET ${encoderWasmUrl} failed: HTTP ${res.status}`);
      const wasmBinary = await res.arrayBuffer();
      const { default: draco3d } = await import('draco3dgltf');
      const encoder = await draco3d.createEncoderModule({ wasmBinary });
      if (typeof encoder?.Encoder !== 'function') throw new Error('the Draco module has no Encoder');
      return { ok: true, encoder };
    } catch (err) {
      return { ok: false, reason: err?.message ?? String(err) };
    }
  })();
  return pending;
}
