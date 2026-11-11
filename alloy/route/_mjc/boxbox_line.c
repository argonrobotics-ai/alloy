/* Line contacts for box-box pairs on MuJoCo C (BUL-0014): alloy.contact.mjw_boxbox_line in float64.
 *
 * MuJoCo's mjc_BoxBox picks the separating axis of least overlap; when an edge-cross axis wins it emits one
 * closest-point contact. When that single contact's two support features along its normal are an edge on a face, or
 * two parallel edges, the overlap segment's two endpoints replace it (the rule and constants of box_box_line).
 * Anything else (a vertex, crossing edges, a segment shorter than SEGMENT_MIN of the boxes' scale, or MuJoCo's
 * multi-contact face-face result) keeps MuJoCo's contacts.
 *
 * The hook sits in mjCOLLISIONFUNC[BOX][BOX] once per process and calls the routine it replaced; the line rule runs
 * only on a thread inside a tick whose config has line = 1 (thread-local), so every other MuJoCo C user keeps
 * mjc_BoxBox bit for bit. MuJoCo calls collision functions with the band margin + gap (engine_collision_driver.c),
 * the band box_box_line_wrapper uses on MuJoCo Warp. Box-box pairs hold up to 8 contacts (mj_maxContact).
 */
#include <math.h>
#include <pthread.h>
#include <string.h>

#include "mjc_internal.h"

#define FEATURE_SIN 0.05    /* |n . axis| below this: the axis lies in the contact plane (about 2.9 deg) */
#define SEGMENT_MIN 1.0e-4  /* shortest segment that becomes two contacts, fraction of the summed half sizes */

typedef struct {
  double pos[3], rot[9], size[3];   /* rot: MuJoCo's row-major geom_xmat; axis k is column k */
} box_t;

typedef struct {
  double span[2], base[3], e[3], f[3], cf[3];
} edge_face_t;

static mjfCollision g_orig = NULL;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local int t_line = 0;
static _Thread_local alloy_mjc_out* t_out = NULL;

static double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

static void axis(const box_t* b, int k, double* a) {
  a[0] = b->rot[k], a[1] = b->rot[3 + k], a[2] = b->rot[6 + k];
}

static void axpy3(double* out, const double* x, double s, const double* y) {   /* out = x + s y */
  for (int i = 0; i < 3; i++) {
    out[i] = x[i] + s * y[i];
  }
}

/* support feature in direction d: kind 0 vertex, 1 edge along *ax, 2 face with normal *ax */
static int feature(const box_t* b, const double* d, int* ax) {
  int n_perp = 0, perp = -1, other = -1;
  for (int k = 0; k < 3; k++) {
    double a[3];
    axis(b, k, a);
    if (fabs(dot3(a, d)) < FEATURE_SIN) {
      n_perp++, perp = k;
    } else {
      other = k;
    }
  }
  *ax = n_perp == 1 ? perp : n_perp == 2 ? other : -1;
  return n_perp == 1 ? 1 : n_perp == 2 ? 2 : 0;
}

/* centre of the box's edge along axis a that is furthest in direction d */
static void support_edge(const box_t* b, const double* d, int a, double* out) {
  memcpy(out, b->pos, sizeof(double) * 3);
  for (int j = 0; j < 3; j++) {
    if (j == a) {
      continue;
    }
    double r[3];
    axis(b, j, r);
    axpy3(out, out, (dot3(r, d) >= 0.0 ? 1.0 : -1.0) * b->size[j], r);
  }
}

/* restrict span to alpha + beta t <= bound (empty: span[0] > span[1]) */
static void clip(double* span, double alpha, double beta, double bound) {
  if (fabs(beta) < 1.0e-12) {
    if (alpha > bound) {
      span[0] = 1.0, span[1] = -1.0;
    }
    return;
  }
  double t = (bound - alpha) / beta;
  if (beta > 0.0) {
    span[1] = fmin(span[1], t);
  } else {
    span[0] = fmax(span[0], t);
  }
}

/* the edge (axis ea of the edge box) clipped against the face (normal axis fa of the face box, facing the edge box)
 * and the band height <= margin */
static void edge_on_face(const box_t* face, int fa, const double* toward, const box_t* edge, int ea, double margin,
                         edge_face_t* o) {
  double r[3], u[3], v[3], w[3], away[3] = {-toward[0], -toward[1], -toward[2]};
  axis(face, fa, r);
  double sg = dot3(r, toward) >= 0.0 ? 1.0 : -1.0;
  for (int i = 0; i < 3; i++) {
    o->f[i] = sg * r[i];
  }
  axpy3(o->cf, face->pos, face->size[fa], o->f);
  axis(face, (fa + 1) % 3, u);
  axis(face, (fa + 2) % 3, v);
  double hu = face->size[(fa + 1) % 3], hv = face->size[(fa + 2) % 3];
  support_edge(edge, away, ea, o->base);
  axis(edge, ea, o->e);
  for (int i = 0; i < 3; i++) {
    w[i] = o->base[i] - o->cf[i];
  }
  o->span[0] = -edge->size[ea], o->span[1] = edge->size[ea];
  clip(o->span, dot3(w, u), dot3(o->e, u), hu);
  clip(o->span, -dot3(w, u), -dot3(o->e, u), hu);
  clip(o->span, dot3(w, v), dot3(o->e, v), hv);
  clip(o->span, -dot3(w, v), -dot3(o->e, v), hv);
  clip(o->span, dot3(w, o->f), dot3(o->e, o->f), margin);
}

static int line_edge_face(const box_t* b1, const box_t* b2, int f1, int a1, int a2, const double* n, double margin,
                          double seg_min, mjPreContact* con) {
  edge_face_t o;
  double sign = 1.0, mn[3] = {-n[0], -n[1], -n[2]};
  if (f1 == 2) {   /* face of box1, edge of box2: the face normal points box1 -> box2 */
    edge_on_face(b1, a1, n, b2, a2, margin, &o);
  } else {         /* face of box2, edge of box1 */
    edge_on_face(b2, a2, mn, b1, a1, margin, &o);
    sign = -1.0;
  }
  if (o.span[1] - o.span[0] <= seg_min) {
    return 0;
  }
  for (int j = 0; j < 2; j++) {
    double q[3], dq[3];
    axpy3(q, o.base, o.span[j], o.e);
    for (int i = 0; i < 3; i++) {
      dq[i] = q[i] - o.cf[i];
    }
    con[j].dist = dot3(dq, o.f);
    axpy3(con[j].pos, q, -0.5 * con[j].dist, o.f);
    for (int i = 0; i < 3; i++) {
      con[j].normal[i] = sign * o.f[i];
    }
  }
  return 2;
}

static int line_edges(const box_t* b1, const box_t* b2, int a1, int a2, const double* n, double margin,
                      double seg_min, mjPreContact* con) {
  double e[3], e2[3], mn[3] = {-n[0], -n[1], -n[2]}, p1[3], p2[3];
  axis(b1, a1, e);
  axis(b2, a2, e2);
  double c = dot3(e, e2);
  if (fabs(c) < sqrt(1.0 - FEATURE_SIN * FEATURE_SIN)) {
    return 0;   /* crossing edges: MuJoCo's closest point is the contact */
  }
  if (c < 0.0) {
    e2[0] = -e2[0], e2[1] = -e2[1], e2[2] = -e2[2];
  }
  support_edge(b1, n, a1, p1);
  support_edge(b2, mn, a2, p2);
  double c1 = dot3(p1, e), c2 = dot3(p2, e);
  double span[2] = {fmax(c1 - b1->size[a1], c2 - b2->size[a2]), fmin(c1 + b1->size[a1], c2 + b2->size[a2])};
  double s0v[3], s1v[3];
  for (int i = 0; i < 3; i++) {   /* separation along n at position t: s0 + s1 t */
    s0v[i] = p2[i] - c2 * e2[i] - p1[i] + c1 * e[i];
    s1v[i] = e2[i] - e[i];
  }
  clip(span, dot3(s0v, n), dot3(s1v, n), margin);
  if (span[1] - span[0] <= seg_min) {
    return 0;
  }
  for (int j = 0; j < 2; j++) {
    double pa[3], pb[3], dp[3];
    axpy3(pa, p1, span[j] - c1, e);
    axpy3(pb, p2, span[j] - c2, e2);
    for (int i = 0; i < 3; i++) {
      dp[i] = pb[i] - pa[i];
      con[j].pos[i] = 0.5 * (pa[i] + pb[i]);
      con[j].normal[i] = n[i];
    }
    con[j].dist = dot3(dp, n);
  }
  return 2;
}

static void box_of(const mjModel* m, const mjData* d, int g, box_t* b) {
  memcpy(b->pos, d->geom_xpos + 3 * g, sizeof(double) * 3);
  memcpy(b->rot, d->geom_xmat + 9 * g, sizeof(double) * 9);
  memcpy(b->size, m->geom_size + 3 * g, sizeof(double) * 3);
}

int mjc_line_apply(const mjModel* m, const mjData* d, int g1, int g2, mjtNum margin, mjPreContact* con) {
  box_t b1, b2;
  box_of(m, d, g1, &b1);
  box_of(m, d, g2, &b2);
  double n[3], mn[3];
  memcpy(n, con[0].normal, sizeof(double) * 3);
  mn[0] = -n[0], mn[1] = -n[1], mn[2] = -n[2];
  int a1, a2;
  int f1 = feature(&b1, n, &a1), f2 = feature(&b2, mn, &a2);
  double seg_min = SEGMENT_MIN * (b1.size[0] + b1.size[1] + b1.size[2] + b2.size[0] + b2.size[1] + b2.size[2]);
  mjPreContact out[2];
  memset(out, 0, sizeof(out));   /* tangent zero, as mjc_BoxBox */
  int k = 0;
  if ((f1 == 2 && f2 == 1) || (f1 == 1 && f2 == 2)) {
    k = line_edge_face(&b1, &b2, f1, a1, a2, n, margin, seg_min, out);
  } else if (f1 == 1 && f2 == 1) {
    k = line_edges(&b1, &b2, a1, a2, n, margin, seg_min, out);
  }
  if (k == 2) {
    memcpy(con, out, sizeof(out));
  }
  return k;
}

static int boxbox_hook(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin) {
  int n = g_orig(m, d, con, g1, g2, margin);
  if (n != 1 || !t_line) {
    return n;
  }
  if (mjc_line_apply(m, d, g1, g2, margin, con) == 2) {
    if (t_out) {
      t_out->line_fired++;
    }
    return 2;
  }
  return n;
}

static void install_hook(void) {
  if (mjCOLLISIONFUNC[mjGEOM_BOX][mjGEOM_BOX] == boxbox_hook) {
    return;
  }
  pthread_mutex_lock(&g_lock);
  if (mjCOLLISIONFUNC[mjGEOM_BOX][mjGEOM_BOX] != boxbox_hook) {
    g_orig = mjCOLLISIONFUNC[mjGEOM_BOX][mjGEOM_BOX];
    mjCOLLISIONFUNC[mjGEOM_BOX][mjGEOM_BOX] = boxbox_hook;
  }
  pthread_mutex_unlock(&g_lock);
}

void mjc_line_begin(const alloy_mjc_config* cfg, alloy_mjc_out* out) {
  if (!cfg->line) {
    return;
  }
  install_hook();
  t_line = 1;
  t_out = out;
}

void mjc_line_end(void) {
  t_line = 0;
  t_out = NULL;
}

void alloy_mjc_unhook(void) {
  pthread_mutex_lock(&g_lock);
  if (g_orig && mjCOLLISIONFUNC[mjGEOM_BOX][mjGEOM_BOX] == boxbox_hook) {
    mjCOLLISIONFUNC[mjGEOM_BOX][mjGEOM_BOX] = g_orig;
  }
  pthread_mutex_unlock(&g_lock);
}

int alloy_mjc_boxbox(const mjModel* m, mjData* d, int32_t g1, int32_t g2, double margin, int32_t line, double* pos,
                     double* normal, double* dist) {
  if (!m || !d || g1 < 0 || g2 < 0 || g1 >= m->ngeom || g2 >= m->ngeom || m->geom_type[g1] != mjGEOM_BOX ||
      m->geom_type[g2] != mjGEOM_BOX || !pos || !normal || !dist) {
    return ALLOY_MJC_E_ARGS;
  }
  mjPreContact con[mjMAXCONPAIR];
  memset(con, 0, sizeof(con));
  mjfCollision boxbox = mjCOLLISIONFUNC[mjGEOM_BOX][mjGEOM_BOX];   /* MuJoCo's, also while hooked */
  int n = (boxbox == boxbox_hook ? g_orig : boxbox)(m, d, con, g1, g2, margin);
  if (n == 1 && line) {
    n = mjc_line_apply(m, d, g1, g2, margin, con) == 2 ? 2 : 1;
  }
  for (int i = 0; i < n; i++) {
    memcpy(pos + 3 * i, con[i].pos, sizeof(double) * 3);
    memcpy(normal + 3 * i, con[i].normal, sizeof(double) * 3);
    dist[i] = con[i].dist;
  }
  return n;
}
