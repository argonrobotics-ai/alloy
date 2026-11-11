// Sheets (settings, fork), the command log drawer, and toasts.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export function openSheet(node) {
  const body = $('sheetBody');
  body.innerHTML = '';
  body.appendChild(node);
  $('sheet').classList.add('open');
  $('scrim').classList.add('on');
}

export function closeSheet() {
  $('sheet').classList.remove('open');
  if (!$('drawer').classList.contains('open')) $('scrim').classList.remove('on');
  $('drawer').classList.remove('open');
  $('scrim').classList.remove('on');
}

const fmt = (v) => (Math.abs(v) >= 100 || v === 0 ? String(+v.toFixed(1)) : Math.abs(v) < 0.01 ? v.toExponential(1) : String(+v.toPrecision(3)));

export function settingsSheet(settings, onSet) {
  const box = document.createElement('div');
  box.innerHTML = '<h3>Settings</h3>';
  for (const s of settings) {
    const log = s.lo > 0 && s.hi / s.lo >= 100;
    const toUnit = (v) => (log ? Math.log(v / s.lo) / Math.log(s.hi / s.lo) : (v - s.lo) / (s.hi - s.lo));
    const fromUnit = (u) => (log ? s.lo * Math.pow(s.hi / s.lo, u) : s.lo + u * (s.hi - s.lo));
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `<label>${esc(s.label)}</label><input type="range" min="0" max="1000"><output>${fmt(s.value)}</output>`;
    const input = row.querySelector('input'), out = row.querySelector('output');
    input.value = Math.round(toUnit(s.value) * 1000);
    input.oninput = () => { out.textContent = fmt(fromUnit(input.value / 1000)); };
    input.onchange = () => {
      const v = +fromUnit(input.value / 1000).toPrecision(3);
      s.value = v;
      onSet(s.name, v);
      toast(`${s.label}: ${fmt(v)}`);
    };
    box.appendChild(row);
  }
  return box;
}

export function forkSheet(S, onChoose) {
  const box = document.createElement('div');
  const w = S.world;
  const worker = S.ov.workers.find((x) => w >= x.offset && w < x.offset + x.worlds);
  const target = pickSlot(S, worker, w);
  box.innerHTML = '<h3>Fork</h3>';
  const choices = [
    [`Branch into world ${target + 1}`, 'That world continues from this moment', { op: 'fork', args: { world: w, into: target } }, `World ${target + 1} now branches from here`],
    ['Save a snapshot', 'Keep this moment to restore later', { op: 'fork', args: { world: w, into: `snapshot:w${w + 1}-${new Date().toTimeString().slice(0, 8)}` } }, 'Snapshot saved'],
  ];
  for (const name of [...new Set(S.snapshots)].slice(-3).reverse()) {
    choices.push([`Restore ${name}`, 'Load that snapshot into this world', { op: 'restore', args: { snapshot: name, world: w } }, `Restored ${name}`]);
  }
  for (const [title, sub, args, label] of choices) {
    const b = document.createElement('button');
    b.className = 'choice';
    b.innerHTML = `${esc(title)}<small>${esc(sub)}</small>`;
    b.onclick = () => onChoose(args, label);
    box.appendChild(b);
  }
  return box;
}

function pickSlot(S, worker, w) {
  const order = S.ov.ret.map((r, i) => [r, i]).filter(([, i]) => i >= worker.offset && i < worker.offset + worker.worlds && i !== w);
  order.sort((a, b) => a[0] - b[0]);
  return order.length ? order[0][1] : w;       // the slot with the lowest current return
}

const ago = (t) => { const s = Math.max(0, Date.now() / 1000 - t); return s < 60 ? `${Math.round(s)} s ago` : `${Math.round(s / 60)} min ago`; };

function describe(e) {
  const a = e.args || {};
  const where = e.gworld !== undefined ? ` world ${e.gworld + 1}` : a.world === null || a.world === undefined ? ' run' : '';
  if (e.op === 'set') return `Set ${a.name} to ${a.value}`;
  if (e.op === 'push') return `Push ${a.body}${where}`;
  if (e.op === 'fork') return String(a.into).startsWith('snapshot:') ? `Snapshot${where}` : `Fork${where}`;
  if (e.op === 'restore') return `Restore ${a.snapshot} into${where}`;
  return `${e.op[0].toUpperCase()}${e.op.slice(1)}${where}`;
}

const who = (e) => (e.issuer || 'user').replace('agent:', 'agent ');

export function logLine(e) { return `${describe(e)} · ${who(e)} · ${ago(e.wall)}`; }

export function renderLog(el, rows) {
  const seen = new Set();
  const uniq = rows.filter((e) => (seen.has(e.id) ? false : seen.add(e.id)));
  el.innerHTML = uniq.slice(-100).reverse().map((e) => {
    const r = e.result || {};
    const res = e.ok ? (e.op === 'set' ? `${r.old} → ${r.new}` : e.op === 'push' ? `${r.force} N for ${r.steps} steps` : '') : (r.error || 'failed');
    return `<div class="entry${e.ok ? '' : ' fail'}"><span><b>${esc(describe(e))}</b></span>
      <span class="who${(e.issuer || '').startsWith('agent') ? ' agent' : ''}">${esc(who(e))}</span>
      <span class="res">step ${e.step} · ${new Date(e.wall * 1000).toLocaleTimeString()}${res ? ' · ' + esc(res) : ''}</span></div>`;
  }).join('');
}

let toastTimer = null;
export function toast(text) {
  const t = $('toast');
  t.textContent = text;
  t.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('on'), 2200);
}
