// Play-Edward engine: personal opening book in front of Maia-2 (ONNX, in-browser).
import { boardToTensor, mirrorFen, mirrorMove, N_CHANNELS } from "./maia-encode.js";

// Shipped in three parts: each stays under GitHub's 25 MB web-upload cap, so the
// whole site can be published by drag-and-drop without the git command line.
const MODEL_PARTS = ["./maia2.fp16.onnx.part0", "./maia2.fp16.onnx.part1",
                     "./maia2.fp16.onnx.part2"];
const META_URL = "./model_meta.json";
const BOOK_URL = "./book.json";

export const STYLES = {
  stock: {},
  attacking: { capture: 0.9, check: 1.3, king_approach: 0.7, advance: 0.35 },
  positional: { capture: -0.5, check: -0.4, centre: 0.8, develop: 0.5, advance: -0.2 },
  wild: { capture: 1.6, check: 2.2, king_approach: 1.2, sacrifice: 1.4 },
};

const CENTRE = new Set(["d4", "e4", "d5", "e5", "c4", "f4", "c5", "f5"]);
const VALUE = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };

/**
 * chess.js writes the en-passant square after every double push; python-chess
 * (and therefore the training data and the book) only writes it when an
 * en-passant capture is actually legal. Normalise to the latter, or both the
 * book lookups and the ep input channel will silently disagree with training.
 */
export function normaliseFen(chess) {
  const fen = chess.fen();
  const parts = fen.split(" ");
  if (parts[3] === "-") return fen;
  const hasEp = chess.moves({ verbose: true }).some((m) => m.flags.includes("e"));
  if (!hasEp) parts[3] = "-";
  return parts.join(" ");
}

export const epdKey = (fen) => fen.split(" ").slice(0, 4).join(" ");

export class Engine {
  constructor() {
    this.session = null; this.meta = null; this.book = null;
    this.moveIndex = null; this.ready = false;
  }

  /** Book and metadata are tiny — load them first so play can start at once. */
  async loadLight() {
    const [meta, book] = await Promise.all([
      fetch(META_URL).then((r) => r.json()),
      fetch(BOOK_URL).then((r) => r.json()),
    ]);
    this.meta = meta;
    this.book = book.book ?? book;
    this.moveIndex = new Map(meta.moves.map((m, i) => [m, i]));
    return this;
  }

  async loadModel(onProgress) {
    const ort = window.ort;
    // GitHub Pages cannot set COOP/COEP headers, so SharedArrayBuffer is
    // unavailable and multi-threaded WASM would fail. Single thread is correct
    // here; WebGPU does the heavy lifting where it is available.
    ort.env.wasm.numThreads = 1;
    // wasm paths resolve relative to ort.min.js, which already sits in vendor/
    const buf = await fetchParts(MODEL_PARTS, onProgress);
    this.session = await ort.InferenceSession.create(buf, {
      executionProviders: ["webgpu", "wasm"], graphOptimizationLevel: "all",
    });
    this.ready = true;
    return this;
  }

  bookMove(fen, chess) {
    const e = this.book[epdKey(fen)];
    if (!e) return null;
    const legal = new Set(chess.moves({ verbose: true }).map(uciOf));
    const options = e.moves.filter(([m]) => legal.has(m));
    if (!options.length) return null;
    const total = options.reduce((s, [, w]) => s + w, 0);
    let r = Math.random() * total;
    for (const [m, w] of options) { r -= w; if (r <= 0) return m; }
    return options[0][0];
  }

  /** emb_self = (1-alpha) * player vector + alpha * elo_embedding[bin] */
  embedding(alpha, bin) {
    const v = this.meta.player_vector;
    const e = this.meta.elo_embedding[bin ?? this.meta.elo_embedding.length - 1];
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = (1 - alpha) * v[i] + alpha * e[i];
    return out;
  }

  async modelMove(fen, chess, { style = "stock", temperature = 0.9, alpha = 0 } = {}) {
    const ort = window.ort;
    const white = fen.split(" ")[1] === "w";
    const persp = white ? fen : mirrorFen(fen);
    const boards = boardToTensor(persp);

    const out = await this.session.run({
      boards: new ort.Tensor("float32", boards, [1, N_CHANNELS * 64]),
      emb_self: new ort.Tensor("float32", this.embedding(alpha, this.meta.his_bin), [1, 128]),
      elos_oppo: new ort.Tensor("int64", BigInt64Array.from([BigInt(this.meta.his_bin)]), [1]),
    });
    const logits = out.policy.data;

    // Legal moves, expressed in the model's (possibly mirrored) frame.
    const legal = chess.moves({ verbose: true }).map((m) => {
      const uci = uciOf(m);
      return { uci, model: white ? uci : mirrorMove(uci), san: m.san, raw: m };
    });

    const weights = STYLES[style] ?? {};
    const scored = [];
    for (const mv of legal) {
      const idx = this.moveIndex.get(mv.model);
      if (idx === undefined) continue;
      let z = logits[idx] + styleBias(mv.raw, weights, white);
      scored.push({ ...mv, z });
    }
    if (!scored.length) return legal[0]?.uci ?? null;

    const t = Math.max(0.05, temperature);
    const max = Math.max(...scored.map((s) => s.z));
    let sum = 0;
    for (const s of scored) { s.p = Math.exp((s.z - max) / t); sum += s.p; }
    let r = Math.random() * sum;
    for (const s of scored) { r -= s.p; if (r <= 0) return s.uci; }
    return scored[0].uci;
  }

  async chooseMove(chess, opts) {
    const fen = normaliseFen(chess);
    const b = this.bookMove(fen, chess);
    if (b) return { uci: b, source: "book" };
    if (!this.ready) return { uci: null, source: "waiting" };
    return { uci: await this.modelMove(fen, chess, opts), source: "model" };
  }
}

function uciOf(m) { return m.from + m.to + (m.promotion ?? ""); }

/** Features are computed from the mover's perspective, so "forward" is up. */
function styleBias(m, weights, white) {
  if (!Object.keys(weights).length) return 0;
  const rank = (sq) => (white ? Number(sq[1]) : 9 - Number(sq[1]));
  const f = {
    capture: m.flags.includes("c") || m.flags.includes("e") ? 1 : 0,
    check: m.san.includes("+") || m.san.includes("#") ? 1 : 0,
    centre: CENTRE.has(m.to) ? 1 : 0,
    advance: Math.max(0, (rank(m.to) - rank(m.from)) / 7),
    develop: rank(m.from) <= 2 && rank(m.to) >= 3 ? 1 : 0,
    king_approach: 0,
    sacrifice: (m.captured && VALUE[m.piece] > VALUE[m.captured]) ? 1 : 0,
  };
  let z = 0;
  for (const [k, w] of Object.entries(weights)) z += w * (f[k] ?? 0);
  return z;
}

/** Fetch the parts in order and glue them back into one ArrayBuffer. */
async function fetchParts(urls, onProgress) {
  const buffers = [];
  let done = 0;
  for (const u of urls) {
    buffers.push(new Uint8Array(await fetchWithProgress(u, (f) =>
      onProgress?.((done + f) / urls.length))));
    done += 1;
    onProgress?.(done / urls.length);
  }
  const total = buffers.reduce((s, b) => s + b.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const b of buffers) { out.set(b, off); off += b.length; }
  return out.buffer;
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return await res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onProgress?.(got / total);
  }
  const buf = new Uint8Array(got); let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf.buffer;
}
