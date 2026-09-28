/**
 * Minimal lossless PNG encoder (8-bit grayscale or RGBA), for the texture
 * embedded in an exported .glb.
 *
 * In plain words: a PNG is a signature plus chunks (IHDR size/format, IDAT
 * zlib-compressed rows, IEND). Each row is written with filter type 0 (none),
 * so the pixels are stored exactly. zlib compression uses the platform's
 * CompressionStream('deflate') -- present in browsers and in Node >= 18 --
 * so no dependency is needed. No colour management, no gamma chunk.
 */

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const COLOR_TYPE = { gray: 0, rgba: 6 };
const CHANNELS = { gray: 1, rgba: 4 };

let crcTable = null;
function crc32(bytes) {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = crcTable[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** zlib (RFC 1950) stream of `bytes`. */
export async function zlibDeflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * @param {{width:number, height:number, data:Uint8Array, format:'gray'|'rgba'}} img  row-major, first row = top
 * @returns {Promise<Uint8Array>}
 */
export async function encodePng({ width, height, data, format }) {
  const channels = CHANNELS[format];
  if (!channels) throw new Error(`unsupported PNG format ${format}`);
  if (data.length !== width * height * channels) throw new Error(`PNG data has ${data.length} bytes, expected ${width * height * channels}`);
  const stride = width * channels;
  const raw = new Uint8Array((stride + 1) * height);
  for (let r = 0; r < height; r++) raw.set(data.subarray(r * stride, (r + 1) * stride), r * (stride + 1) + 1); // filter byte 0
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, width);
  v.setUint32(4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = COLOR_TYPE[format];
  const parts = [Uint8Array.from(SIGNATURE), chunk('IHDR', ihdr), chunk('IDAT', await zlibDeflate(raw)), chunk('IEND', new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
