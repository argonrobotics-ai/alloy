/* Scoped noslip on MuJoCo C data (SCR-WP8-6): alloy.contact.noslip's algorithm in float64.
 *
 * MuJoCo's noslip post-solve (engine_solver.c solNoSlip) on the frictional rows the scope selects: the same rows,
 * order (efc order: dof friction rows, then contacts), projections (cone.c), cost changes and stopping rule as
 * alloy.contact.noslip.ScopedNoslip. Slots: one dof friction row, or one elliptic contact's dim - 1 friction rows.
 * The sweep runs in qacc space: residual J qacc - aref, then qacc += M^-1 J' delta (Y rows from mj_solveM), each on
 * its row's nonzeros only (a dof row's J is one entry; Y lives on the slot's trees). At the end efc_force,
 * qfrc_constraint (+ J' delta) and qacc are written.
 *
 * restart = 0 (the route's default) starts from the main solve's qacc; restart = 1 first sets the touched trees'
 * qacc = qacc_smooth + M^-1 qfrc_constraint, as MuJoCo's noslip does. Unlike ScopedNoslip there is no tree
 * blocking: rows are full nv (flex contacts included in scope "all"); efc_state is not updated (as ScopedNoslip).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mjc_internal.h"

typedef struct {
  int r0;     /* the dof friction row, or the contact's normal row */
  int nf;     /* friction rows; 0 = a dof friction row */
  int first;  /* first friction efc row (r0 for a dof row) */
  int g;      /* its first row in the scoped row list */
  int con;    /* contact id; -1 for a dof row */
} slot_t;

typedef struct {
  int n, nrow, nv;
  slot_t* slot;    /* [n] */
  int* jnz;        /* [nrow][nv] nonzero dofs of each J row, jcnt[g] of them */
  int* ynz;        /* [nrow][nv] nonzero dofs of each Y row, ycnt[g] of them */
  int* jcnt;       /* [nrow] */
  int* ycnt;       /* [nrow] */
  double* J;       /* [nrow][nv] */
  double* Y;       /* [nrow][nv] M^-1 J' */
  double* blk;     /* [nrow][MJC_MAXNF] the slot's own block of J M^-1 J' */
  double* f;       /* [nrow] current forces */
  double* f0;      /* [nrow] start forces */
  double* lam;     /* [n] condim-6 QCQP multipliers */
  double* q;       /* [nv] qacc */
  double* tmp;     /* [nv] */
} scratch_t;

static int in_pairs(const alloy_mjc_config* c, int lo, int hi) {
  for (int p = 0; p < c->npair; p++) {
    if (c->pairs[2 * p] == lo && c->pairs[2 * p + 1] == hi) {
      return 1;
    }
  }
  return 0;
}

/* 1 if efc row r starts a scoped slot (alloy.contact.noslip_kernels.keep_row) */
static int keep_row(const alloy_mjc_config* c, const mjData* d, int r) {
  int type = d->efc_type[r];
  if (type == mjCNSTR_FRICTION_DOF) {
    return c->dof_scope[d->efc_id[r]] != 0;
  }
  if (type != mjCNSTR_CONTACT_ELLIPTIC) {
    return 0;
  }
  const mjContact* con = d->contact + d->efc_id[r];
  if (con->efc_address != r || con->dim <= 1) {
    return 0;
  }
  int g0 = con->geom[0], g1 = con->geom[1];
  int sa = g0 >= 0 ? c->geom_scope[g0] : 0, sb = g1 >= 0 ? c->geom_scope[g1] : 0;
  int keep = c->noslip_all + ((sa ^ sb) & ALLOY_MJC_ROBOT_BIT) + ((sa ^ sb) & ALLOY_MJC_SURF_BIT);
  if (keep == 0 && (sa & sb & ALLOY_MJC_PAIR_BIT)) {
    keep = in_pairs(c, g0 < g1 ? g0 : g1, g0 < g1 ? g1 : g0);
  }
  return keep > 0;
}

static int slot_rows(const mjData* d, int r) {
  return d->efc_type[r] == mjCNSTR_FRICTION_DOF ? 1 : d->contact[d->efc_id[r]].dim - 1;
}

static void dense_row(const mjModel* m, const mjData* d, int r, double* out) {
  int nv = m->nv;
  if (mj_isSparse(m)) {
    memset(out, 0, sizeof(double) * nv);
    int adr = d->efc_J_rowadr[r];
    for (int p = 0; p < d->efc_J_rownnz[r]; p++) {
      out[d->efc_J_colind[adr + p]] = d->efc_J[adr + p];
    }
  } else {
    memcpy(out, d->efc_J + (size_t)r * nv, sizeof(double) * nv);
  }
}

/* a . b over the nonzeros idx[0..n) of a */
static double sdot(const double* a, const int* idx, int n, const double* b) {
  double s = 0.0;
  for (int p = 0; p < n; p++) {
    s += a[idx[p]] * b[idx[p]];
  }
  return s;
}

/* x += s y over the nonzeros idx[0..n) of y */
static void saxpy(double* x, const double* y, const int* idx, int n, double s) {
  for (int p = 0; p < n; p++) {
    x[idx[p]] += y[idx[p]] * s;
  }
}

static int nonzeros(const double* row, int nv, int* idx) {
  int n = 0;
  for (int j = 0; j < nv; j++) {
    if (row[j] != 0.0) {
      idx[n++] = j;
    }
  }
  return n;
}

static int alloc_scratch(scratch_t* s, int n, int nrow, int nv) {
  size_t nd = (size_t)nrow * nv * 2 + (size_t)nrow * MJC_MAXNF + 2 * (size_t)nrow + n + 2 * (size_t)nv;
  double* buf = (double*)calloc(nd, sizeof(double));
  slot_t* slot = (slot_t*)calloc((size_t)n, sizeof(slot_t));
  int* ibuf = (int*)malloc(sizeof(int) * ((size_t)nrow * nv * 2 + 2 * (size_t)nrow));
  if (!buf || !slot || !ibuf) {
    free(buf);
    free(slot);
    free(ibuf);
    return 0;
  }
  s->n = n, s->nrow = nrow, s->nv = nv, s->slot = slot;
  s->jnz = ibuf;
  s->ynz = s->jnz + (size_t)nrow * nv;
  s->jcnt = s->ynz + (size_t)nrow * nv;
  s->ycnt = s->jcnt + nrow;
  s->J = buf;
  s->Y = s->J + (size_t)nrow * nv;
  s->blk = s->Y + (size_t)nrow * nv;
  s->f = s->blk + (size_t)nrow * MJC_MAXNF;
  s->f0 = s->f + nrow;
  s->lam = s->f0 + nrow;
  s->q = s->lam + n;
  s->tmp = s->q + nv;
  return 1;
}

static void free_scratch(scratch_t* s) {
  free(s->J);
  free(s->slot);
  free(s->jnz);
}

/* slots in efc order, their dense rows, Y = M^-1 J' and each slot's own block */
static void gather(const alloy_mjc_config* c, const mjModel* m, const mjData* d, scratch_t* s) {
  int k = 0, g = 0, nv = m->nv;
  for (int r = 0; r < d->nefc && k < s->n; r++) {
    if (!keep_row(c, d, r)) {
      continue;
    }
    slot_t* sl = s->slot + k++;
    int dof = d->efc_type[r] == mjCNSTR_FRICTION_DOF;
    sl->r0 = r, sl->nf = dof ? 0 : slot_rows(d, r), sl->first = dof ? r : r + 1, sl->g = g;
    sl->con = dof ? -1 : d->efc_id[r];
    for (int i = 0; i < (dof ? 1 : sl->nf); i++, g++) {
      dense_row(m, d, sl->first + i, s->J + (size_t)g * nv);
      s->jcnt[g] = nonzeros(s->J + (size_t)g * nv, nv, s->jnz + (size_t)g * nv);
      s->f[g] = s->f0[g] = d->efc_force[sl->first + i];
    }
  }
  mj_solveM(m, (mjData*)d, s->Y, s->J, s->nrow);
  for (int r = 0; r < s->nrow; r++) {
    s->ycnt[r] = nonzeros(s->Y + (size_t)r * nv, nv, s->ynz + (size_t)r * nv);
  }
  for (int i = 0; i < s->n; i++) {
    const slot_t* sl = s->slot + i;
    int nr = sl->nf ? sl->nf : 1;
    for (int a = 0; a < nr; a++) {
      size_t ga = (size_t)(sl->g + a);
      for (int b = 0; b < nr; b++) {
        s->blk[ga * MJC_MAXNF + b] = sdot(s->J + ga * nv, s->jnz + ga * nv, s->jcnt[ga], s->Y + (size_t)(sl->g + b) * nv);
      }
    }
  }
}

static double row_dot_q(const scratch_t* s, int g) {
  size_t o = (size_t)g * s->nv;
  return sdot(s->J + o, s->jnz + o, s->jcnt[g], s->q);
}

static void q_add_row(scratch_t* s, int g, double delta) {
  size_t o = (size_t)g * s->nv;
  saxpy(s->q, s->Y + o, s->ynz + o, s->ycnt[g], delta);
}

static int tree_of_geom(const mjModel* m, int g) {
  return g >= 0 ? m->body_treeid[m->geom_bodyid[g]] : -1;
}

/* MuJoCo's start on the trees the slots touch: qacc = qacc_smooth + M^-1 qfrc_constraint */
static int restart_trees(const mjModel* m, mjData* d, const scratch_t* s) {
  int ntree = m->ntree > 0 ? m->ntree : 1;
  unsigned char* need = (unsigned char*)calloc((size_t)ntree, 1);
  if (!need) {
    return 0;
  }
  for (int i = 0; i < s->n; i++) {
    const slot_t* sl = s->slot + i;
    if (sl->con < 0) {
      need[m->dof_treeid[d->efc_id[sl->r0]]] = 1;
      continue;
    }
    for (int j = 0; j < 2; j++) {
      int t = tree_of_geom(m, d->contact[sl->con].geom[j]);
      if (t >= 0) {
        need[t] = 1;
      }
    }
  }
  mj_solveM(m, d, s->tmp, d->qfrc_constraint, 1);
  for (int j = 0; j < m->nv; j++) {
    if (need[m->dof_treeid[j]]) {
      d->qacc[j] = d->qacc_smooth[j] + s->tmp[j];
    }
  }
  free(need);
  return 1;
}

/* MuJoCo adds the rows' 0.5 f^2 R to the first sweep's improvement (here: the slots' rows, normal row included) */
static double regularizer_cost(const mjData* d, const scratch_t* s) {
  double c = 0.0;
  for (int i = 0; i < s->n; i++) {
    const slot_t* sl = s->slot + i;
    for (int k = 0; k <= sl->nf; k++) {
      double D = d->efc_D[sl->r0 + k];
      if (D > 0.0) {
        c += 0.5 * d->efc_force[sl->r0 + k] * d->efc_force[sl->r0 + k] / D;
      }
    }
  }
  return c;
}

static double sweep_dof(const mjData* d, scratch_t* s, const slot_t* sl) {
  int g = sl->g;
  double r = row_dot_q(s, g) - d->efc_aref[sl->r0];
  double adiag = fmax(mjMINVAL, s->blk[(size_t)g * MJC_MAXNF]);
  double change, old = s->f[g];
  s->f[g] = mjc_dof_force(adiag, d->efc_frictionloss[sl->r0], r, old, &change);
  q_add_row(s, g, s->f[g] - old);
  return change;
}

static double sweep_contact(const mjData* d, scratch_t* s, int i, double la0) {
  const slot_t* sl = s->slot + i;
  int g = sl->g, nf = sl->nf;
  double Ac[MJC_MAXNF * MJC_MAXNF], res[MJC_MAXNF], old[MJC_MAXNF], force[MJC_MAXNF], la = 0.0;
  for (int k = 0; k < nf; k++) {
    for (int l = 0; l < nf; l++) {
      Ac[k * nf + l] = s->blk[(size_t)(g + k) * MJC_MAXNF + l];
    }
    Ac[k * nf + k] = fmax(1.0e-10, Ac[k * nf + k]);
    res[k] = row_dot_q(s, g + k) - d->efc_aref[sl->first + k];
    old[k] = s->f[g + k];
  }
  double change = mjc_contact_block(Ac, res, old, d->contact[sl->con].friction, d->efc_force[sl->r0], nf, la0,
                                    force, &la);
  s->lam[i] = la;
  for (int k = 0; k < nf; k++) {
    s->f[g + k] = force[k];
    q_add_row(s, g + k, force[k] - old[k]);
  }
  return change;
}

/* the stop test's scale: 1 / trace(M) at the current configuration, as MuJoCo 3.17's own noslip scales it (3.16:
 * trace(M(qpos0)) = stat.meaninertia * nv); d->M is the reduced sparse inertia, each row's diagonal its last entry */
static double inertia_scale(const mjModel* m, const mjData* d) {
  double trace = 0.0;
  for (int i = 0; i < m->nv; i++) {
    trace += d->M[m->M_rowadr[i] + m->M_rownnz[i] - 1];
  }
  return m->nv > 0 ? 1.0 / (trace > mjMINVAL ? trace : mjMINVAL) : 1.0;
}

/* projected Gauss-Seidel sweeps; returns the number of sweeps */
static int sweeps(const alloy_mjc_config* c, const mjModel* m, const mjData* d, scratch_t* s) {
  double scale = inertia_scale(m, d);
  double reg = regularizer_cost(d, s);
  int it = 0;
  while (it < c->noslip_iterations) {
    double improvement = it == 0 ? reg : 0.0;
    int warm = it > 0 && c->noslip_warm_qcqp;
    for (int i = 0; i < s->n; i++) {
      const slot_t* sl = s->slot + i;
      improvement -= sl->nf == 0 ? sweep_dof(d, s, sl) : sweep_contact(d, s, i, warm ? s->lam[i] : 0.0);
    }
    it++;
    if (improvement * scale < c->noslip_tolerance) {
      break;
    }
  }
  return it;
}

static void apply(const mjModel* m, mjData* d, const scratch_t* s) {
  int nv = m->nv;
  for (int i = 0; i < s->n; i++) {
    const slot_t* sl = s->slot + i;
    for (int k = 0; k < (sl->nf ? sl->nf : 1); k++) {
      int g = sl->g + k;
      d->efc_force[sl->first + k] = s->f[g];
      saxpy(d->qfrc_constraint, s->J + (size_t)g * nv, s->jnz + (size_t)g * nv, s->jcnt[g], s->f[g] - s->f0[g]);
    }
  }
  memcpy(d->qacc, s->q, sizeof(double) * nv);
}

static void count_slots(const alloy_mjc_config* c, const mjData* d, int* n, int* nrow, int* over) {
  *n = *nrow = *over = 0;
  for (int r = 0; r < d->nefc; r++) {
    if (!keep_row(c, d, r)) {
      continue;
    }
    if (*n >= c->noslip_max_slots) {
      *over = 1;   /* the rest keep the main solve's forces */
      continue;
    }
    (*n)++;
    *nrow += slot_rows(d, r);
  }
}

int mjc_noslip_run(const alloy_mjc_config* c, const mjModel* m, mjData* d, alloy_mjc_out* out) {
  int n, nrow, over;
  if (d->nefc == 0 || m->nv == 0) {
    return ALLOY_MJC_OK;
  }
  count_slots(c, d, &n, &nrow, &over);
  if (out) {
    out->noslip_slots += n;
    out->noslip_overflow += over;
  }
  if (n == 0) {
    return ALLOY_MJC_OK;
  }
  scratch_t s;
  if (!alloc_scratch(&s, n, nrow, m->nv)) {
    return ALLOY_MJC_E_NOMEM;
  }
  gather(c, m, d, &s);
  if (c->noslip_restart && !restart_trees(m, d, &s)) {
    free_scratch(&s);
    return ALLOY_MJC_E_NOMEM;
  }
  memcpy(s.q, d->qacc, sizeof(double) * m->nv);
  int it = sweeps(c, m, d, &s);
  apply(m, d, &s);
  d->solver_niter[0] += it;   /* as MuJoCo's monolithic noslip and ScopedNoslip */
  if (out) {
    out->noslip_sweeps += it;
  }
  free_scratch(&s);
  return ALLOY_MJC_OK;
}
