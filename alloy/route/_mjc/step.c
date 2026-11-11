/* The CPU route's tick (SCR-WP8-6): MuJoCo C steps with Alloy's per-step plug-ins, contact export, state pull and
 * health counters (ABI: alloy_mjc.h).
 *
 * One step is mj_step1, then mj_step2's work for the Euler and implicit integrators with the scoped noslip after the
 * constraint solve (MuJoCo 3.16 engine_forward.c mj_step2: mj_effActuation and the metric terms run only for the
 * discrete integrator, which is refused here with RK4; IPC belongs to the discrete integrator). With the plug-ins
 * off a tick equals mj_step bit for bit (tests/route/test_mjc_step.py). The integrator saves qacc_warmstart from the
 * post-noslip qacc (mj_advance).
 */
#include <math.h>
#include <string.h>

#include "mjc_internal.h"

int alloy_mjc_abi(void) { return ALLOY_MJC_ABI; }

int alloy_mjc_mujoco_version(void) { return mjVERSION_HEADER; }

void alloy_mjc_struct_sizes(int32_t* sizes) {
  sizes[0] = (int32_t)sizeof(alloy_mjc_config);
  sizes[1] = (int32_t)sizeof(alloy_mjc_out);
}

static int check_noslip(const alloy_mjc_config* c, const mjModel* m) {
  if (c->noslip_iterations <= 0) {
    return ALLOY_MJC_OK;
  }
  if (m->opt.cone != mjCONE_ELLIPTIC) {
    return ALLOY_MJC_E_CONE;
  }
  if (c->noslip_max_slots <= 0 || !c->geom_scope || !c->dof_scope || (c->npair > 0 && !c->pairs)) {
    return ALLOY_MJC_E_ARGS;
  }
  return ALLOY_MJC_OK;
}

static int check_tick(const alloy_mjc_config* c, const mjModel* m, const mjData* d, int32_t nstep,
                      const alloy_mjc_out* out) {
  if (!c || !m || !d || !out || nstep < 0) {
    return ALLOY_MJC_E_ARGS;
  }
  if (c->abi != ALLOY_MJC_ABI) {
    return ALLOY_MJC_E_ABI;
  }
  int integ = m->opt.integrator;
  if (integ != mjINT_EULER && integ != mjINT_IMPLICIT && integ != mjINT_IMPLICITFAST) {
    return ALLOY_MJC_E_INTEGRATOR;
  }
  if (m->opt.enableflags & mjENBL_SLEEP) {
    return ALLOY_MJC_E_SLEEP;
  }
  if (c->noslip_iterations > 0 && m->opt.noslip_iterations > 0) {
    return ALLOY_MJC_E_NOSLIP_TWICE;
  }
  if (d->threadpool) {   /* collision on pool threads would miss the thread-local line flag */
    return ALLOY_MJC_E_ARGS;
  }
  int bodies = c->pull_state || (out->traj_cap > 0 && out->traj_body_q);
  if (bodies && (c->nbody_out <= 0 || !c->body_map)) {
    return ALLOY_MJC_E_ARGS;
  }
  if (c->pull_state && (!out->body_q || !out->body_qd)) {
    return ALLOY_MJC_E_ARGS;
  }
  if (out->traj_cap > 0 && (out->traj_cap < nstep || !out->traj_body_q || !out->traj_body_qd)) {
    return ALLOY_MJC_E_ARGS;
  }
  if (c->export_contacts && out->ncon_max > 0 &&
      (!out->geom || !out->pos || !out->frame || !out->dist || !out->force)) {
    return ALLOY_MJC_E_ARGS;
  }
  for (int i = 0; bodies && i < c->nbody_out; i++) {
    if (c->body_map[i] >= m->nbody) {
      return ALLOY_MJC_E_ARGS;
    }
  }
  return check_noslip(c, m);
}

/* body_q [nbody_out][7], body_qd [nbody_out][6] in body_map order (FK first: the step left xpos stale) */
static void pull_state(const alloy_mjc_config* c, const mjModel* m, mjData* d, double* q, double* qd) {
  mj_kinematics(m, d);
  mj_comPos(m, d);
  mj_comVel(m, d);
  for (int i = 0; i < c->nbody_out; i++) {
    int b = c->body_map[i];
    if (b < 0) {
      continue;
    }
    double* qi = q + 7 * i;
    double* vi = qd + 6 * i;
    double v[6];
    memcpy(qi, d->xpos + 3 * b, sizeof(double) * 3);
    qi[3] = d->xquat[4 * b + 1], qi[4] = d->xquat[4 * b + 2], qi[5] = d->xquat[4 * b + 3], qi[6] = d->xquat[4 * b];
    mj_objectVelocity(m, d, mjOBJ_BODY, b, v, 0);   /* [ang, lin] at the COM, world orientation */
    memcpy(vi, v + 3, sizeof(double) * 3);
    memcpy(vi + 3, v, sizeof(double) * 3);
  }
}

static void export_contacts(const mjModel* m, const mjData* d, alloy_mjc_out* out) {
  int n = d->ncon < out->ncon_max ? d->ncon : out->ncon_max;
  for (int i = 0; i < n; i++) {
    const mjContact* con = d->contact + i;
    out->geom[2 * i] = con->geom[0];
    out->geom[2 * i + 1] = con->geom[1];
    memcpy(out->pos + 3 * i, con->pos, sizeof(double) * 3);
    memcpy(out->frame + 9 * i, con->frame, sizeof(double) * 9);
    out->dist[i] = con->dist;
    mj_contactForce(m, d, i, out->force + 6 * i);
  }
  out->ncon = n;
  if (d->ncon > n) {
    out->contact_overflow += d->ncon - n;
  }
}

static int bad_warnings(const mjData* d) {
  return d->warning[mjWARN_BADQPOS].number + d->warning[mjWARN_BADQVEL].number +
         d->warning[mjWARN_BADQACC].number;
}

/* per-step counters: non-finite or reset steps, deepest contact, fastest dof */
static void step_counters(const mjModel* m, const mjData* d, int bad_before, alloy_mjc_out* out) {
  int bad = bad_warnings(d) != bad_before;
  double vmax = 0.0, pen = 0.0;
  for (int i = 0; i < m->nq; i++) {
    bad |= !isfinite(d->qpos[i]);
  }
  for (int i = 0; i < m->nv; i++) {
    bad |= !isfinite(d->qvel[i]) || !isfinite(d->qacc[i]);
    vmax = fmax(vmax, fabs(d->qvel[i]));
  }
  for (int i = 0; i < d->ncon; i++) {
    pen = fmax(pen, -d->contact[i].dist);
  }
  out->steps++;
  out->nonfinite += bad;
  out->qvel_max = fmax(out->qvel_max, vmax);
  out->pen_max = fmax(out->pen_max, pen);
}

/* one physics step; returns ALLOY_MJC_OK or ALLOY_MJC_E_NOMEM (that step's noslip skipped) */
static int one_step(const alloy_mjc_config* c, const mjModel* m, mjData* d, alloy_mjc_out* out) {
  int err = ALLOY_MJC_OK, bad_before = bad_warnings(d);
  mj_step1(m, d);
  mj_fwdActuation(m, d);
  mj_fwdAcceleration(m, d);
  mj_fwdConstraint(m, d);
  if (c->noslip_iterations > 0) {
    err = mjc_noslip_run(c, m, d, out);
  }
  d->flg_rnepost = 0;
  mj_sensorAcc(m, d);
  mj_checkAcc(m, d);
  if (m->opt.enableflags & mjENBL_FWDINV) {
    mj_compareFwdInv(m, d);
  }
  if (m->opt.integrator == mjINT_EULER) {
    mj_Euler(m, d);
  } else {
    mj_implicit(m, d);
  }
  step_counters(m, d, bad_before, out);
  return err;
}

int alloy_mjc_tick(const alloy_mjc_config* c, const mjModel* m, mjData* d, int32_t nstep, alloy_mjc_out* out) {
  int err = check_tick(c, m, d, nstep, out), first = ALLOY_MJC_OK;
  if (err) {
    return err;
  }
  int traj = out->traj_cap > 0 && out->traj_body_q;
  mjc_line_begin(c, out);
  for (int k = 0; k < nstep; k++) {
    err = one_step(c, m, d, out);
    first = first ? first : err;
    if (traj) {
      pull_state(c, m, d, out->traj_body_q + (size_t)k * c->nbody_out * 7,
                 out->traj_body_qd + (size_t)k * c->nbody_out * 6);
    }
  }
  mjc_line_end();
  if (c->export_contacts && out->ncon_max > 0) {
    export_contacts(m, d, out);
  }
  if (c->pull_state) {
    pull_state(c, m, d, out->body_q, out->body_qd);
  }
  return first;
}

int alloy_mjc_tick_batch(int32_t nworld, const alloy_mjc_config* const* c, const mjModel* const* m,
                         mjData* const* d, const uint8_t* active, int32_t nstep, alloy_mjc_out* out,
                         int32_t* status) {
  if (nworld < 0 || !c || !m || !d || !out || !status) {
    return nworld > 0 ? nworld : 0;
  }
  int failed = 0;
  for (int w = 0; w < nworld; w++) {
    status[w] = (active && !active[w]) ? ALLOY_MJC_OK : alloy_mjc_tick(c[w], m[w], d[w], nstep, out + w);
    failed += status[w] != ALLOY_MJC_OK;
  }
  return failed;
}

int alloy_mjc_noslip(const alloy_mjc_config* c, const mjModel* m, mjData* d, alloy_mjc_out* out) {
  if (!c || !m || !d) {
    return ALLOY_MJC_E_ARGS;
  }
  if (c->abi != ALLOY_MJC_ABI) {
    return ALLOY_MJC_E_ABI;
  }
  if (m->opt.enableflags & mjENBL_SLEEP) {
    return ALLOY_MJC_E_SLEEP;
  }
  int err = check_noslip(c, m);
  return err ? err : mjc_noslip_run(c, m, d, out);
}
