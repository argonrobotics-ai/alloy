// Alloy Live: one page, two views (overview grid of every world; one world with video, 3D, timeline, commands).
import { VideoView, parsePacket } from './video.js';
import { View3D } from './view3d.js';
import { openSheet, closeSheet, settingsSheet, forkSheet, renderLog, logLine, toast } from './ui.js';

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const params = new URLSearchParams(location.search);
const S = { ov: null, world: null, live: true, geo: {}, log: [], past: null, flags: {}, snapshots: [], pausedWorlds: new Set() };

// --- connection -----------------------------------------------------------------------------------------------------
let ws;
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => (typeof e.data === 'string' ? onText(JSON.parse(e.data)) : onBinary(e.data));
  ws.onopen = () => { if (S.world !== null) send({ type: 'open', world: S.world }); };
  ws.onclose = () => { $('sub').textContent = 'reconnecting'; setTimeout(connect, 1000); };
}
const send = (m) => ws && ws.readyState === 1 && ws.send(JSON.stringify(m));
export function command(op, args = {}) { send({ type: 'command', op, args, issuer: 'user' }); }

function onText(m) {
  if (m.type === 'overview') { S.ov = m; S.flags = m.flags || {}; drawHeader(); if (S.live || S.world === null) drawGrid(m.status); updateTimeline(); }
  else if (m.type === 'past') { S.past = m; drawGrid(m.status, true); if (!S.live && S.world !== null) setScrub(m.t); }
  else if (m.type === 'geometry') { S.geo[m.world] = m.geo; if (m.world === S.world) view3d.setGeometry(m.geo, robotsOf(m.world)); }
  else if (m.type === 'state') { if (m.world === S.world && (S.live || m.past || m.play)) view3d.setState(m); }
  else if (m.type === 'log') { S.log.push(m.entry); onEntry(m.entry); }
  else if (m.type === 'log_all') { S.log = m.entries; refreshLog(); }
  else if (m.type === 'error') toast(m.error);
  else if (m.type === 'ended') goLive();
}

function onBinary(buf) {
  const p = parsePacket(buf);
  if (p.type === 2 && S.world !== null) video.push(p.data, p.t, p.key, S.live, p.meta);
}

// --- header ---------------------------------------------------------------------------------------------------------
function drawHeader() {
  const ov = S.ov;
  $('run').textContent = S.world === null ? ov.run : `World ${S.world + 1}`;
  const machines = ov.workers.filter((w) => w.connected).length;
  const paused = ov.workers.length && ov.workers.every((w) => w.paused);
  $('sub').textContent = S.world === null
    ? `${ov.worlds} worlds · ${machines} machine${machines === 1 ? '' : 's'}${paused ? ' · paused' : ''}`
    : `${ov.run} · machine ${machineOf(S.world) + 1}`;
  $('sps').textContent = ov.steps_per_s >= 1000 ? `${(ov.steps_per_s / 1000).toFixed(1)}k` : Math.round(ov.steps_per_s);
  $('runPause').classList.toggle('on', paused);
  $('ret').textContent = ov.line.length ? Math.round(ov.line[ov.line.length - 1][1]) : '-';
  drawSpark(ov.line, ov.marks);
}

function drawSpark(line, marks) {
  const svg = $('spark');
  if (!line || line.length < 2) { svg.innerHTML = ''; return; }
  const t0 = line[0][0], t1 = line[line.length - 1][0];
  const ys = line.map((p) => p[1]);
  const lo = Math.min(...ys), hi = Math.max(...ys), span = hi - lo || 1;
  const X = (t) => ((t - t0) / (t1 - t0 || 1)) * 100, Y = (v) => 31 - ((v - lo) / span) * 28;
  const d = line.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(2)},${Y(p[1]).toFixed(2)}`).join('');
  const ticks = (marks || []).filter((m) => m.op === 'set' && m.t >= t0 && m.t <= t1)
    .map((m) => `<line class="mark" x1="${X(m.t)}" x2="${X(m.t)}" y1="0" y2="34"/>`).join('');
  svg.innerHTML = `${ticks}<path class="area" d="${d}L100,34L0,34Z"/><path class="line" d="${d}" vector-effect="non-scaling-stroke"/>`;
}

// --- overview grid ----------------------------------------------------------------------------------------------------
const robotsOf = (gw) => (S.ov && S.ov.workers[machineOf(gw)]?.robots) || [];
const machineOf = (gw) => S.ov.workers.findIndex((w) => gw >= w.offset && gw < w.offset + w.worlds);

function buildGrid() {
  const root = $('machines');
  root.innerHTML = '';
  S.ov.workers.forEach((w, k) => {
    const m = document.createElement('div');
    m.className = 'machine';
    m.innerHTML = `<div class="mlabel">Machine ${k + 1} · ${w.worlds} worlds</div><div class="grid"></div>`;
    const grid = m.querySelector('.grid');
    for (let i = 0; i < w.worlds; i++) {
      const sq = document.createElement('div');
      sq.className = 'sq';
      sq.dataset.w = w.offset + i;
      sq.onclick = () => openWorld(w.offset + i);
      grid.appendChild(sq);
    }
    root.appendChild(m);
  });
  root.dataset.key = S.ov.workers.map((w) => `${w.name}:${w.worlds}`).join(',');
}

function drawGrid(status, past = false) {
  if (!S.ov) return;
  const key = S.ov.workers.map((w) => `${w.name}:${w.worlds}`).join(',');
  if ($('machines').dataset.key !== key) buildGrid();
  $('overview').classList.toggle('past', past);
  document.querySelectorAll('.sq').forEach((sq) => {
    const gw = +sq.dataset.w;
    sq.className = `sq s${status[gw] ?? 4}${S.flags[gw] && !past ? ' flag' : ''}`;
  });
}

// --- world view ---------------------------------------------------------------------------------------------------------
const video = new VideoView($('video'));
if (!video.supported) $('novideo').hidden = false;      // WebCodecs: Safari 16.4+ / Chrome, over https
const view3d = new View3D($('view3d'), {
  accent: css('--accent'), robot: '#7b8494', body: '#9aa3b2', ground: css('--neutral'), cloth: css('--cloth'),
});
view3d.onPush = (body, force) => {
  command('push', { world: S.world, body, force, steps: 10 });
  setPushMode(false);                     // one push per tap of Push
  toast(`Pushed the ${body.startsWith('cable') ? 'cable' : body} with ${Math.hypot(...force).toFixed(1)} N`);
};

function openWorld(gw) {
  S.world = gw; S.live = true;
  $('overview').hidden = true; $('world').hidden = false; $('back').hidden = false; $('runPause').hidden = true;
  const why = { best: 'Highest return', worst: 'Lowest return', alarm: 'Health alarm' };
  $('wtag').textContent = (S.flags[gw] || []).map((f) => why[f] || f).join(' · ');
  video.clear();
  video.world = gw;
  if (S.geo[gw]) view3d.setGeometry(S.geo[gw], robotsOf(gw));
  send({ type: 'open', world: gw });
  history.replaceState(null, '', `?world=${gw}${params.has('debug') ? '&debug=1' : ''}`);
  setPushMode(false);
  drawHeader(); refreshLog(); updateTimeline();
  view3d.resize();
}

function closeWorld() {
  S.world = null; S.live = true;
  send({ type: 'close' });
  $('overview').hidden = false; $('world').hidden = true; $('back').hidden = true; $('runPause').hidden = false;
  history.replaceState(null, '', location.pathname);
  drawHeader(); drawGrid(S.ov.status);
}

function setPushMode(on) {
  view3d.setPushMode(on);
  document.querySelector('[data-cmd=push]').classList.toggle('on', on);
  $('ptag').hidden = !on;
  if (on) $('media').classList.add('show3d');
}

const worldPaused = () => S.ov && S.ov.status[S.world] === 3;

function onCommandButton(cmd) {
  if (cmd === 'pause') command(worldPaused() ? 'resume' : 'pause', { world: S.world });
  else if (cmd === 'push') setPushMode(!view3d.pushMode);
  else if (cmd === 'reset') command('reset', { world: S.world });
  else if (cmd === 'fork') openSheet(forkSheet(S, (args, label) => { command(args.op, args.args); toast(label); closeSheet(); }));
  else if (cmd === 'settings') openSheet(settingsSheet(S.ov.workers[0]?.settings || [], (name, value) => command('set', { name, value })));
}

function onEntry(e) {
  refreshLog();
  if (e.op === 'fork' && String(e.args.into).startsWith('snapshot:')) S.snapshots.push(String(e.args.into).slice(9));
  const mine = e.gworld === S.world || e.gworld === undefined;
  if (e.issuer !== 'user' && mine) toast(`${e.issuer.replace('agent:', 'Agent ')} sent ${e.op}`);
  if (!e.ok && e.issuer === 'user') toast(e.result?.error || 'Command failed');
}

function refreshLog() {
  const rows = S.world === null ? S.log : S.log.filter((e) => e.gworld === undefined || e.gworld === S.world || e.op === 'set');
  $('logline').textContent = rows.length ? logLine(rows[rows.length - 1]) : 'No commands yet';
  renderLog($('log'), rows);
  document.querySelector('[data-cmd=pause]').textContent = worldPaused() ? 'Resume' : 'Pause';
}

// --- timeline -----------------------------------------------------------------------------------------------------------
let seekTimer = null;
function span() { return S.ov && S.ov.line.length ? [S.ov.line[0][0] - 5, S.ov.t] : null; }

function updateTimeline() {
  if (S.world === null || !S.ov) return;
  document.querySelector('[data-cmd=pause]').textContent = worldPaused() ? 'Resume' : 'Pause';
  const sp = span();
  if (!sp) return;
  const [t0, t1] = sp;
  const marks = S.ov.marks.filter((m) => m.t && (m.world === S.world || m.world === null || m.world === undefined));
  $('marks').innerHTML = marks.map((m) => `<i class="${m.issuer && m.issuer.startsWith('agent') ? 'agent' : ''}" style="left:${((m.t - t0) / (t1 - t0)) * 100}%"></i>`).join('');
  if (S.live) $('scrub').value = 1000;
  $('live').hidden = S.live; $('tnow').hidden = !S.live;
  $('tnow').textContent = S.live ? 'live' : '';
}

function setScrub(t) {
  const sp = span();
  if (sp) $('scrub').value = Math.round(((t - sp[0]) / (sp[1] - sp[0])) * 1000);
}

function scrubTime() {
  const [t0, t1] = span();
  return t0 + ($('scrub').value / 1000) * (t1 - t0);
}

$('scrub').addEventListener('input', () => {
  if (!span()) return;
  S.live = false; updateTimeline();
  clearTimeout(seekTimer);
  seekTimer = setTimeout(() => { video.reset(); send({ type: 'seek', world: S.world, t: scrubTime() }); }, 60);
});
$('scrub').addEventListener('change', () => {
  if (!span()) return;
  if (+$('scrub').value >= 998) { goLive(); return; }
  video.reset();
  send({ type: 'play', world: S.world, t: scrubTime() });
});
function goLive() {
  S.live = true; video.reset(); send({ type: 'open', world: S.world }); updateTimeline();
  if (S.ov) drawGrid(S.ov.status);
}
$('live').onclick = goLive;

// --- wiring -----------------------------------------------------------------------------------------------------------
document.querySelectorAll('#bar button').forEach((b) => { b.onclick = () => onCommandButton(b.dataset.cmd); });
$('back').onclick = closeWorld;
$('toggle').onclick = () => { const on = $('media').classList.toggle('show3d'); $('toggle').textContent = on ? 'Camera' : '3D'; view3d.resize(); };
$('runPause').onclick = () => {
  const paused = S.ov && S.ov.workers.every((w) => w.paused);
  command(paused ? 'resume' : 'pause', { world: null });
  toast(paused ? 'Run resumed' : 'Run paused');
};
$('logline').onclick = () => { $('drawer').classList.add('open'); $('scrim').classList.add('on'); };
$('scrim').onclick = () => { $('drawer').classList.remove('open'); closeSheet(); };
if (params.has('debug')) setInterval(() => console.log('video latency median s', video.medianLatency()), 5000);
window.__live = { S, video, view3d, command, openWorld };
connect();
if (params.has('world')) { const gw = +params.get('world'); const wait = setInterval(() => { if (S.ov) { clearInterval(wait); openWorld(gw); } }, 100); }
