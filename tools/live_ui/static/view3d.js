// The 3D view of one world: bodies from the worker's geometry (sent once), posed from the 12 Hz state stream; cloth
// drawn from its streamed points. Z is up, as in Alloy. In push mode a tap on a body and a drag apply a force.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const PUSH_N_PER_M = 30;      // drag length -> force
const PUSH_MAX_N = 8;

export class View3D {
  constructor(el, colors) {
    this.el = el;
    this.colors = colors;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 20);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.46, -0.60, 0.44);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, -0.03, 0.08);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8890a0, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(0.6, -0.4, 1.2);
    this.scene.add(sun);
    this.bodies = [];          // index in the frame's body arrays -> {group, name}
    this.robots = [];
    this.cloths = {};
    this.pushMode = false;
    this.onPush = null;
    this._drag = null;
    this._ray = new THREE.Raycaster();
    this._bindPointer();
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => { this.controls.update(); this.renderer.render(this.scene, this.camera); requestAnimationFrame(loop); };
    loop();
  }

  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  clear() {
    for (const b of this.bodies) this.scene.remove(b.group);
    for (const c of Object.values(this.cloths)) for (const m of c.shells) this.scene.remove(m);
    if (this.statics) this.scene.remove(this.statics);
    this.bodies = []; this.cloths = {}; this.statics = null;
  }

  _material(name, isStatic) {
    const c = this.colors;
    const robot = this.robots.some((r) => name === r || name.startsWith(`${r}/`));
    const color = isStatic ? c.ground : robot ? c.robot : name === 'cube' ? c.accent
      : name.startsWith('cable') ? c.cloth : c.body;
    return new THREE.MeshStandardMaterial({ color, roughness: 0.7, metalness: 0.0 });
  }

  // One shape of the frame export's topology (sizes are full, as in alloy.Scene; pose in its body or the world).
  _shape(s, mat) {
    let g;
    if (s.kind === 'box' || s.kind === 'ellipsoid') g = s.kind === 'box' ? new THREE.BoxGeometry(...s.size)
      : new THREE.SphereGeometry(0.5, 24, 16).scale(...s.size);
    else if (s.kind === 'sphere') g = new THREE.SphereGeometry(s.radius, 24, 16);
    else if (s.kind === 'capsule') g = new THREE.CapsuleGeometry(s.radius, s.height, 6, 16).rotateX(Math.PI / 2);
    else if (s.kind === 'cylinder') g = new THREE.CylinderGeometry(s.radius, s.radius, s.height, 24).rotateX(Math.PI / 2);
    else if (s.kind === 'cone') g = new THREE.ConeGeometry(s.radius, s.height, 24).rotateX(Math.PI / 2);
    else if (s.kind === 'plane') g = new THREE.PlaneGeometry(s.size[0] || 1.2, s.size[1] || 1.2);
    else if ((s.kind === 'mesh' || s.kind === 'convex_mesh') && s.vertices) {
      g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(s.vertices.flat(), 3));
      g.setIndex(s.faces.flat());
      g.computeVertexNormals();
    } else return null;
    const m = new THREE.Mesh(g, mat);
    m.position.set(...s.pos);
    m.quaternion.set(...s.quat_xyzw);
    return m;
  }

  // topo: the frame export's topology part (body_names, body_shapes, statics, mesh_faces, particles, rods).
  setGeometry(topo, robots = []) {
    this.clear();
    this.robots = robots;
    this.statics = new THREE.Group();
    for (const s of topo.statics || []) {
      const m = this._shape(s, this._material('', true));
      if (m) this.statics.add(m);
    }
    this.scene.add(this.statics);
    this.bodies = topo.body_names.map((name, i) => {
      const group = new THREE.Group();
      for (const s of topo.body_shapes[i] || []) {
        const m = this._shape(s, this._material(name, false));
        if (m) { m.userData.body = name; group.add(m); }
      }
      group.visible = false;                       // shown once the first pose arrives
      this.scene.add(group);
      return { group, name };
    });
    const names = new Set([...Object.keys(topo.mesh_faces || {}), ...Object.keys(topo.mesh_tets || {})]);
    for (const name of names) this._addMesh(name, topo);
    // a rod drawn by its own bodies (a MuJoCo chain) needs no tube; a solver rod (no bodies) gets one
    this.tubeRods = new Set(Object.entries(topo.rods || {}).filter(([r, info]) => (info.drawn_by_bodies !== undefined
      ? !info.drawn_by_bodies : !topo.body_names.some((b) => b === r || b.startsWith(`${r}/`)))).map(([r]) => r));
  }

  // frame: one frame of the export (body_pos / body_quat_xyzw [B][3|4], meshes, particles, rods); plain arrays.
  setState(frame) {
    const p = frame.body_pos || [], q = frame.body_quat_xyzw || [];
    for (let i = 0; i < this.bodies.length && i < p.length; i++) {
      const g = this.bodies[i].group;
      g.position.set(...p[i]);
      g.quaternion.set(...q[i]);
      g.visible = true;
    }
    for (const [name, verts] of Object.entries(frame.meshes || {})) this._setMesh(name, verts);
    for (const [name, line] of Object.entries(frame.rods || {})) if (this.tubeRods && this.tubeRods.has(name)) this._rod(name, line.flat());
    for (const [name, set] of Object.entries(frame.particles || {})) {
      const pts = set.pos.filter((_, k) => !set.active || set.active[k]);
      this._points(name, pts.flat());
    }
  }

  // A cloth, film or soft-body mesh (export slice 2). Soft bodies (mesh_tets) draw the outer faces of their active
  // tets, so dormant fragments stay hidden and a cut opens the surface. Sheets draw the faces whose vertices are all
  // active (mesh_vertex_active), as two shells at +-thickness/2 around the exported mid-surface, so a sheet on a
  // table rests on it instead of sinking half its thickness in.
  _addMesh(name, topo) {
    const tets = topo.mesh_tets?.[name], vAct = topo.mesh_vertex_active?.[name];
    let faces, n = vAct ? vAct.length : 0;
    if (tets) {
      faces = boundaryFaces(tets, topo.mesh_tet_active?.[name]);
      n = Math.max(n, 1 + maxIndex(tets));
    } else {
      faces = (topo.mesh_faces[name] || []).filter((f) => !vAct || (vAct[f[0]] && vAct[f[1]] && vAct[f[2]]));
      n = Math.max(n, 1 + maxIndex(topo.mesh_faces[name] || []));
    }
    const h = tets ? 0 : (topo.mesh_thickness?.[name] || 0);
    const mat = new THREE.MeshStandardMaterial({ color: tets ? this.colors.body : this.colors.cloth, side: THREE.DoubleSide,
      roughness: 0.9, flatShading: Boolean(tets), polygonOffset: !tets, polygonOffsetFactor: -1, polygonOffsetUnits: -1 });
    const shells = (h > 0 ? [1, -1] : [0]).map(() => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
      g.setIndex(faces.flat());
      const m = new THREE.Mesh(g, mat);
      m.frustumCulled = false;
      m.visible = false;
      this.scene.add(m);
      return m;
    });
    const mid = new THREE.BufferGeometry();          // the mid-surface, for the shells' normals
    mid.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
    mid.setIndex(faces.flat());
    this.cloths[name] = { shells, mid, n, h, faces: faces.length };
  }

  _setMesh(name, verts) {
    const c = this.cloths[name];
    if (!c || verts.length !== c.n) return;
    const flat = verts.flat();
    if (c.h <= 0) {
      const g = c.shells[0].geometry;
      g.getAttribute('position').array.set(flat);
      g.getAttribute('position').needsUpdate = true;
      g.computeVertexNormals();
      c.shells[0].visible = true;
      return;
    }
    c.mid.getAttribute('position').array.set(flat);
    c.mid.computeVertexNormals();
    const nrm = c.mid.getAttribute('normal').array;
    c.shells.forEach((m, k) => {
      const sign = k === 0 ? 0.5 : -0.5;
      const pos = m.geometry.getAttribute('position');
      for (let i = 0; i < flat.length; i++) pos.array[i] = flat[i] + sign * c.h * nrm[i];
      pos.needsUpdate = true;
      m.geometry.computeVertexNormals();
      m.visible = true;
    });
  }

  _rod(name, pts) {
    const v = [];
    for (let i = 0; i + 2 < pts.length; i += 3) v.push(new THREE.Vector3(pts[i], pts[i + 1], pts[i + 2]));
    if (v.length < 2) return;
    this.rods = this.rods || {};
    const old = this.rods[name];
    if (old) { this.scene.remove(old); old.geometry.dispose(); }
    const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(v), v.length * 3, 0.005, 8),
      new THREE.MeshStandardMaterial({ color: this.colors.cloth, roughness: 0.8 }));
    this.rods[name] = tube;
    this.scene.add(tube);
  }

  _points(name, pts) {
    this.parts = this.parts || {};
    let obj = this.parts[name];
    if (!obj || obj.geometry.getAttribute('position').count !== pts.length / 3) {
      if (obj) this.scene.remove(obj);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(pts.length), 3));
      obj = new THREE.Points(g, new THREE.PointsMaterial({ color: this.colors.accent, size: 0.006 }));
      obj.frustumCulled = false;
      this.parts[name] = obj;
      this.scene.add(obj);
    }
    const attr = obj.geometry.getAttribute('position');
    attr.array.set(pts);
    attr.needsUpdate = true;
  }

  // Client (CSS px) position of a body's centre: used by the clip recorder to drag from it.
  screenOf(name) {
    const b = this.bodies.find((x) => x.name === name);
    if (!b) return null;
    const v = b.group.position.clone().project(this.camera);
    const r = this.renderer.domElement.getBoundingClientRect();
    return { x: r.left + ((v.x + 1) / 2) * r.width, y: r.top + ((1 - v.y) / 2) * r.height };
  }

  setPushMode(on) {
    this.pushMode = on;
    this.controls.enabled = !on;
    this.el.classList.toggle('pushing', on);
  }

  _pointer(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }

  _onPlane(e, z) {
    this._ray.setFromCamera(this._pointer(e), this.camera);
    const p = new THREE.Vector3();
    return this._ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -z), p) ? p : null;
  }

  _bindPointer() {
    const el = this.renderer.domElement;
    el.addEventListener('pointerdown', (e) => {
      if (!this.pushMode) return;
      this._ray.setFromCamera(this._pointer(e), this.camera);
      const hits = this._ray.intersectObjects(this.bodies.map((b) => b.group), true);
      const isRobot = (n) => this.robots.some((r) => n === r || n.startsWith(`${r}/`));
      const hit = hits.find((h) => h.object.userData.body && !isRobot(h.object.userData.body));
      if (!hit) return;
      const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), hit.point, 0.001, this.colors.accent, 0.02, 0.012);
      this.scene.add(arrow);
      this._drag = { body: hit.object.userData.body, from: hit.point.clone(), arrow, force: [0, 0, 0] };
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointermove', (e) => {
      const d = this._drag;
      if (!d) return;
      const p = this._onPlane(e, d.from.z);
      if (!p) return;
      const v = p.sub(d.from);
      const len = v.length();
      if (len < 1e-4) return;
      d.arrow.setDirection(v.clone().normalize());
      d.arrow.setLength(Math.min(len, 0.15), 0.02, 0.012);
      const light = d.body.startsWith('cable') ? 0.12 : 1;     // the cable weighs grams: a gentler pull
      const f = Math.min(len * PUSH_N_PER_M, PUSH_MAX_N) * light;
      d.force = [v.x / len * f, v.y / len * f, 0];
    });
    el.addEventListener('pointerup', () => {
      const d = this._drag;
      if (!d) return;
      this._drag = null;
      setTimeout(() => this.scene.remove(d.arrow), 600);
      if (this.onPush && Math.hypot(...d.force) > 0.02) this.onPush(d.body, d.force.map((x) => +x.toFixed(3)));
    });
  }
}

// Outer faces of the active tets: a face shared by two active tets is inside the body.
export function boundaryFaces(tets, active) {
  const seen = new Map();
  tets.forEach((t, i) => {
    if (active && !active[i]) return;
    const [a, b, c, d] = t;
    for (const f of [[a, b, c], [a, d, b], [b, d, c], [a, c, d]]) {
      const key = [...f].sort((x, y) => x - y).join(',');
      seen.set(key, seen.has(key) ? null : f);
    }
  });
  return [...seen.values()].filter(Boolean);
}

// Largest index in a list of index tuples (a loop: spreading a big mesh into Math.max overflows Safari's arg limit).
function maxIndex(rows) {
  let m = -1;
  for (const r of rows) for (const i of r) if (i > m) m = i;
  return m;
}
