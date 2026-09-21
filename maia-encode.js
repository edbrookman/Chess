// Board encoding for Maia-2, ported from maia2/utils.py:board_to_tensor.
// Parses the FEN directly so it cannot drift from a chess library's semantics.
// Layout: 18 channels of 8x8, flattened as channel*64 + rank*8 + file,
// where rank 0 is rank "1" and file 0 is the a-file (python-chess square order).

export const PIECE_ORDER = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };
export const N_CHANNELS = 18;

export function parseFen(fen) {
  const [placement, turn, castling, ep] = fen.trim().split(/\s+/);
  return { placement, turn, castling, ep };
}

/** Mirror a FEN vertically and swap colours — python-chess Board.mirror(). */
export function mirrorFen(fen) {
  const parts = fen.trim().split(/\s+/);
  const ranks = parts[0].split("/").reverse().map((r) =>
    [...r].map((c) => (/[a-z]/.test(c) ? c.toUpperCase()
                     : /[A-Z]/.test(c) ? c.toLowerCase() : c)).join("")
  );
  const castling = parts[2] === "-" ? "-" :
    [...parts[2]].map((c) => (/[a-z]/.test(c) ? c.toUpperCase() : c.toLowerCase()))
      .sort((a, b) => "KQkq".indexOf(a) - "KQkq".indexOf(b)).join("");
  const ep = parts[3] === "-" ? "-" : parts[3][0] + String(9 - Number(parts[3][1]));
  return [ranks.join("/"), parts[1] === "w" ? "b" : "w", castling, ep,
          parts[4] ?? "0", parts[5] ?? "1"].join(" ");
}

export function mirrorSquare(sq) { return sq[0] + String(9 - Number(sq[1])); }

export function mirrorMove(uci) {
  return mirrorSquare(uci.slice(0, 2)) + mirrorSquare(uci.slice(2, 4)) + uci.slice(4);
}

export function boardToTensor(fen) {
  const { placement, turn, castling, ep } = parseFen(fen);
  const t = new Float32Array(N_CHANNELS * 64);

  // Placement is written from rank 8 down to rank 1.
  const rows = placement.split("/");
  for (let i = 0; i < 8; i++) {
    const rank = 7 - i;                 // rank index 0 == rank "1"
    let file = 0;
    for (const ch of rows[i]) {
      if (/\d/.test(ch)) { file += Number(ch); continue; }
      const type = PIECE_ORDER[ch.toLowerCase()];
      const isWhite = ch === ch.toUpperCase();
      const channel = type + (isWhite ? 0 : 6);
      t[channel * 64 + rank * 8 + file] = 1;
      file++;
    }
  }

  if (turn === "w") t.fill(1, 12 * 64, 13 * 64);          // side to move
  const rights = ["K", "Q", "k", "q"];                     // W-king, W-queen, B-king, B-queen
  rights.forEach((r, i) => {
    if (castling.includes(r)) t.fill(1, (13 + i) * 64, (14 + i) * 64);
  });
  if (ep && ep !== "-") {
    const file = ep.charCodeAt(0) - 97;
    const rank = Number(ep[1]) - 1;
    t[17 * 64 + rank * 8 + file] = 1;
  }
  return t;
}
