// Plain decoding through the original, pinned browser WebAssembly module.
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';

const [wasmPath, encodedPath, outputPath, infoPath] = process.argv.slice(2);
assert(wasmPath && encodedPath && outputPath && infoPath);
const module = await WebAssembly.compile(readFileSync(wasmPath));
const imports = WebAssembly.Module.imports(module);
assert.deepEqual(imports, [{ module: 'env', name: 'emscripten_notify_memory_growth', kind: 'function' }]);
const instance = await WebAssembly.instantiate(module, { env: { emscripten_notify_memory_growth: () => {} } });
const X = instance.exports;
X._initialize();
const encoded = readFileSync(encodedPath);
assert(encoded.length > 8 && encoded.length <= 3 * 1024 ** 2);
const pointer = X.vc_in(encoded.length);
assert(pointer > 0);
new Uint8Array(X.memory.buffer, pointer, encoded.length).set(encoded);
const mode = X.vc_chunk_mode(encoded.length);
const q = X.vc_stream_q(encoded.length);
assert(mode >= 0 && q >= 0);
const start = performance.now();
assert.equal(X.vc_decode(encoded.length), 0);
const decoded = Buffer.from(new Uint8Array(X.memory.buffer, X.vc_out(), 128 ** 3));
assert.equal(X.vc_decode(encoded.length), 0);
assert.deepEqual(Buffer.from(new Uint8Array(X.memory.buffer, X.vc_out(), 128 ** 3)), decoded);
// Independent API address check: one16-cubed block must equal that part of the full decode.
assert.equal(X.vc_decode_block(encoded.length, 6, 2, 5), 0);
const block = new Uint8Array(X.memory.buffer, X.vc_block_out(), 16 ** 3);
for (let z = 0; z < 16; z++) for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
  assert.equal(block[(z * 16 + y) * 16 + x], decoded[((z + 96) * 128 + y + 32) * 128 + x + 80]);
}
function cstr(p) {
  const bytes = new Uint8Array(X.memory.buffer);
  let end = p;
  while (end < bytes.length && bytes[end]) end++;
  assert(end < bytes.length);
  return new TextDecoder().decode(bytes.subarray(p, end));
}
const info = { status: 'PASS', mode, q, version: cstr(X.vc_version()), kernels: cstr(X.vc_kernels()),
  output_bytes: decoded.length, decoded_sha256: createHash('sha256').update(decoded).digest('hex'),
  replay_exact: true, block_api_matches_full: true, smoothing: 0,
  decoder_and_checks_ms: performance.now() - start, wasm_memory_bytes: X.memory.buffer.byteLength };
writeFileSync(outputPath, decoded);
writeFileSync(infoPath, JSON.stringify(info, null, 2));
console.log(JSON.stringify(info));
