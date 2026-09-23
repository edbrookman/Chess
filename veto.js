// Stockfish 10 (classic evaluation, no NNUE file) as a blunder veto.
//
// ONE multi-PV search per move. Stockfish returns its best K moves with
// evaluations; we keep those within `threshold` centipawns of the best and let
// Maia pick among them by its own preference. Nine separate evaluations per
// move was the expensive way to get the same information.
const SF_URL = "./vendor/sf/stockfish.js";

export class Veto {
  constructor() { this.sf = null; this.pending = null; this.lines = []; }

  async load() {
    // stockfish.js IS a worker script; load it directly so it can resolve
    // stockfish.wasm sitting next to it. A blob wrapper breaks that resolution.
    this.sf = new Worker(new URL(SF_URL, import.meta.url));
    this.sf.onmessage = (e) => this._line(typeof e.data === "string" ? e.data : e.data?.data ?? "");
    this.send("uci");
    await this._await(/uciok/);
    this.send("setoption name MultiPV value 12");
    this.send("setoption name Hash value 16");
    this.send("isready");
    await this._await(/readyok/);
    return this;
  }

  send(cmd) { this.sf.postMessage(cmd); }

  _line(l) {
    this.lines.push(l);
    if (this.pending && this.pending.re.test(l)) {
      const p = this.pending; this.pending = null; p.resolve(l);
    }
  }

  _await(re, timeout = 20000) {
    return new Promise((resolve, reject) => {
      this.pending = { re, resolve };
      setTimeout(() => { if (this.pending) { this.pending = null; reject(new Error("engine timeout")); } }, timeout);
    });
  }

  /** @returns {Promise<Array<{uci:string, cp:number}>>} top moves, best first */
  async topMoves(fen, depth) {
    this.lines = [];
    this.send("position fen " + fen);
    this.send("go depth " + depth);
    await this._await(/^bestmove/);
    const best = new Map();
    for (const l of this.lines) {
      const m = l.match(/multipv (\d+).*?score (cp|mate) (-?\d+).*? pv (\S+)/);
      if (!m) continue;
      const cp = m[2] === "mate"
        ? (Number(m[3]) > 0 ? 10000 - Number(m[3]) : -10000 - Number(m[3]))
        : Number(m[3]);
      best.set(Number(m[1]), { uci: m[4], cp });          // later depths overwrite
    }
    return [...best.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
  }
}
