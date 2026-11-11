// H.264 (Annex-B access units from NVENC) -> WebCodecs VideoDecoder -> canvas. Each worker sends ONE mosaic stream
// of its picked worlds (tiles in row-major order); every packet names the world of each tile, and the open world's
// tile is cropped onto the canvas.
// Works in desktop Chrome and iPhone Safari 16.4+ (secure context: https or localhost). Each packet is converted to
// length-prefixed AVCC and the decoder is configured from the stream's own SPS/PPS (avcC description), which both
// browsers accept. A key frame (every second) re-syncs after any drop.

function nalUnits(buf) {
  const out = [];
  let i = 0, start = -1;
  while (i + 3 <= buf.length) {
    const sc3 = buf[i] === 0 && buf[i + 1] === 0 && buf[i + 2] === 1;
    const sc4 = !sc3 && i + 4 <= buf.length && buf[i] === 0 && buf[i + 1] === 0 && buf[i + 2] === 0 && buf[i + 3] === 1;
    if (sc3 || sc4) {
      if (start >= 0) out.push(buf.subarray(start, i));
      i += sc3 ? 3 : 4;
      start = i;
    } else i += 1;
  }
  if (start >= 0) out.push(buf.subarray(start));
  return out;
}

function avcc(nals) {
  const keep = nals.filter((n) => { const t = n[0] & 31; return t !== 7 && t !== 8 && t !== 9; });
  const size = keep.reduce((s, n) => s + 4 + n.length, 0);
  const out = new Uint8Array(size);
  let o = 0;
  for (const n of keep) {
    new DataView(out.buffer).setUint32(o, n.length);
    out.set(n, o + 4);
    o += 4 + n.length;
  }
  return out;
}

function description(sps, pps) {
  const d = new Uint8Array(11 + sps.length + pps.length);
  d.set([1, sps[1], sps[2], sps[3], 0xff, 0xe1, sps.length >> 8, sps.length & 255], 0);
  d.set(sps, 8);
  let o = 8 + sps.length;
  d.set([1, pps.length >> 8, pps.length & 255], o);
  d.set(pps, o + 3);
  return d;
}

const hex = (b) => b.toString(16).padStart(2, '0');

export class VideoView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dec = null;
    this.configured = false;
    this.lastT = 0;
    this.latency = [];
    this.onFrame = null;
    this.world = null;
    this.meta = new Map();            // frame timestamp -> {tile, cols, tiles}
    this.supported = typeof VideoDecoder !== 'undefined';
  }

  reset() {
    if (this.dec && this.dec.state !== 'closed') this.dec.close();
    this.dec = null;
    this.configured = false;
  }

  clear() {
    this.reset();
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  _make() {
    this.dec = new VideoDecoder({
      output: (frame) => {
        const m = this.meta.get(frame.timestamp);
        for (const k of this.meta.keys()) { if (k <= frame.timestamp) this.meta.delete(k); else break; }
        const i = m ? m.tiles.indexOf(this.world) : -1;
        if (i < 0) { frame.close(); return; }
        const size = m.tile * 2;
        if (this.canvas.width !== size) { this.canvas.width = size; this.canvas.height = size; }
        this.ctx.imageSmoothingQuality = 'high';
        this.ctx.drawImage(frame, (i % m.cols) * m.tile, Math.floor(i / m.cols) * m.tile, m.tile, m.tile, 0, 0, size, size);
        const t = frame.timestamp / 1e6;
        this.lastT = t;
        if (this.live) this.latency.push(Date.now() / 1000 - t);
        if (this.latency.length > 300) this.latency.shift();
        frame.close();
        if (this.onFrame) this.onFrame(t);
      },
      error: () => { this.configured = false; },
    });
  }

  // data: Uint8Array Annex-B access unit; t: seconds (worker clock); key: bool; meta: {tile, cols, tiles}
  push(data, t, key, live = true, meta = null) {
    if (!this.supported) return;
    this.live = live;
    if (meta) this.meta.set(Math.round(t * 1e6), meta);
    const nals = nalUnits(data);
    if (!this.configured) {
      if (!key) return;
      const sps = nals.find((n) => (n[0] & 31) === 7);
      const pps = nals.find((n) => (n[0] & 31) === 8);
      if (!sps || !pps) return;
      if (!this.dec || this.dec.state === 'closed') this._make();
      this.dec.configure({
        codec: `avc1.${hex(sps[1])}${hex(sps[2])}${hex(sps[3])}`,
        description: description(sps, pps),
        optimizeForLatency: true,
      });
      this.configured = true;
    }
    if (this.dec.decodeQueueSize > 30) { this.reset(); return; }
    try {
      this.dec.decode(new EncodedVideoChunk({ type: key ? 'key' : 'delta', timestamp: Math.round(t * 1e6), data: avcc(nals) }));
    } catch (e) {
      this.configured = false;
    }
  }

  medianLatency() {
    if (!this.latency.length) return null;
    const s = [...this.latency].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
  }
}

// Binary packet from the hub: u8 type (2), f64 t, u8 key, u16 tile px, u8 cols, u8 n, n x u32 world, access unit.
export function parsePacket(buf) {
  const v = new DataView(buf);
  const n = v.getUint8(13);
  const tiles = [];
  for (let i = 0; i < n; i++) tiles.push(v.getUint32(14 + 4 * i));
  return { type: v.getUint8(0), t: v.getFloat64(1), key: v.getUint8(9) === 1,
           meta: { tile: v.getUint16(10), cols: v.getUint8(12), tiles }, data: new Uint8Array(buf, 14 + 4 * n) };
}
