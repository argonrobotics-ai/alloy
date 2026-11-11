/* Alloy's MuJoCo C extension for the CPU route (SCR-WP8-6): the ABI agreed with WP8.
 *
 * Agreed text: [internal note] notes/wp8-results/cpu-item11/ABI.md (v2). Owner WP2. WP8 calls it from
 * alloy/route/mjc_core*.py through ctypes (alloy.route._mjc.load()).
 *
 * Built against the pinned MuJoCo (3.16.0) headers of the installed wheel. A plain C11 shared library: no Python.h,
 * no OpenMP, sequential over worlds (one world's run is bitwise repeatable).
 */
#ifndef ALLOY_MJC_H_
#define ALLOY_MJC_H_

#include <stdint.h>

#include <mujoco/mujoco.h>

#define ALLOY_MJC_ABI 1

/* return codes: checked before a tick's first step, so nothing has run when one comes back */
#define ALLOY_MJC_OK 0
#define ALLOY_MJC_E_ABI (-1)          /* cfg->abi != ALLOY_MJC_ABI */
#define ALLOY_MJC_E_INTEGRATOR (-2)   /* RK4 or discrete: no mid-step plug-ins there (WP8 uses mj_step) */
#define ALLOY_MJC_E_NOSLIP_TWICE (-3) /* scoped noslip asked while the model's own opt.noslip_iterations > 0 */
#define ALLOY_MJC_E_CONE (-4)         /* scoped noslip needs the elliptic cone */
#define ALLOY_MJC_E_SLEEP (-5)        /* sleeping is not supported */
#define ALLOY_MJC_E_ARGS (-6)         /* NULL pointer, bad sizes, a MuJoCo thread pool bound to d, ... */
#define ALLOY_MJC_E_NOMEM (-7)        /* scratch allocation failed (mid-tick: the step's noslip was skipped) */

/* noslip geom-scope bits (alloy.contact.noslip_kernels) */
#define ALLOY_MJC_ROBOT_BIT 1
#define ALLOY_MJC_PAIR_BIT 2
#define ALLOY_MJC_SURF_BIT 4

typedef struct {
  int32_t abi;                 /* = ALLOY_MJC_ABI */
  /* scoped noslip: alloy.contact.noslip.NoslipSpec, scope arrays from build_scope(); iterations 0 = off */
  int32_t noslip_iterations;
  double noslip_tolerance;
  int32_t noslip_all;          /* 1: scope "all" (every dof friction row and elliptic contact) */
  int32_t noslip_restart;      /* NoslipSpec.restart (route: 0, keep the main solve's qacc) */
  int32_t noslip_warm_qcqp;    /* NoslipSpec.warm_qcqp (route: 1) */
  int32_t noslip_max_slots;    /* rows_per_world, resolved (> 0); extra slots keep the main solve's forces */
  const int32_t* geom_scope;   /* [ngeom] NoslipScope.geom bits */
  const int32_t* dof_scope;    /* [nv] NoslipScope.dof: 1 = the dof's friction row is scoped */
  int32_t npair;
  const int32_t* pairs;        /* [npair][2] NoslipScope.pairs, (min, max) */
  /* box-box line contacts (alloy.contact.mjw_boxbox_line, BUL-0014): 1 = on, 0 = MuJoCo's mjc_BoxBox */
  int32_t line;
  /* what a tick writes into alloy_mjc_out */
  int32_t export_contacts;     /* 0 off, 1 after the tick's last step */
  int32_t pull_state;          /* 1: body_q / body_qd after the tick */
  int32_t nbody_out;           /* rows of body_q / body_qd / traj_* (Newton's per-world body count) */
  const int32_t* body_map;     /* [nbody_out] MuJoCo body id per Newton body row; -1 = leave the row unwritten */
} alloy_mjc_config;

typedef struct {
  /* contact export, capacity ncon_max */
  int32_t ncon_max;
  int32_t ncon;                /* written, <= ncon_max (the rest counted in contact_overflow) */
  int32_t* geom;               /* [ncon_max][2] mjContact.geom */
  double* pos;                 /* [ncon_max][3] world */
  double* frame;               /* [ncon_max][9] mjContact.frame: normal (geom 1 -> geom 2), then the tangents */
  double* dist;                /* [ncon_max] */
  double* force;               /* [ncon_max][6] mj_contactForce: contact frame, normal first */
  /* state (pull_state), body_map order */
  double* body_q;              /* [nbody_out][7] x y z qx qy qz qw (Newton transform order) */
  double* body_qd;             /* [nbody_out][6] COM linear velocity, then angular velocity, world frame */
  int32_t traj_cap;            /* rows in traj_* (>= nstep when used); 0 = no trajectory */
  double* traj_body_q;         /* [traj_cap][nbody_out][7]: step k of the tick at row k (NULL = off) */
  double* traj_body_qd;        /* [traj_cap][nbody_out][6] */
  /* counters, accumulated over ticks and updated every step; Python zeroes them */
  int64_t steps;
  int64_t nonfinite;           /* steps that ended non-finite or that MuJoCo reset (bad qpos/qvel/qacc warning) */
  int64_t contact_overflow;    /* contacts beyond ncon_max in exports */
  double pen_max;              /* max -mjContact.dist over every step's contacts */
  double qvel_max;             /* max |qvel| at the end of each step */
  int64_t noslip_slots, noslip_sweeps, noslip_overflow;
  int64_t line_fired;          /* box-box pairs whose single contact became two */
} alloy_mjc_out;

int alloy_mjc_abi(void);
int alloy_mjc_mujoco_version(void);

/* One tick of one world: nstep physics steps, each
 *   mj_step1 (box-box hook active if cfg->line) | mj_fwdActuation | mj_fwdAcceleration | mj_fwdConstraint |
 *   scoped noslip | mj_sensorAcc | mj_checkAcc | mj_Euler or mj_implicit (mj_step2 for these integrators),
 * then the contact export and state pull. Returns ALLOY_MJC_OK or an ALLOY_MJC_E_* code. */
int alloy_mjc_tick(const alloy_mjc_config* cfg, const mjModel* m, mjData* d, int32_t nstep, alloy_mjc_out* out);

/* Every world with active[w] != 0 (active NULL: all), in order; status[w] gets its code (0 if skipped).
 * Returns the number of worlds with a nonzero status. */
int alloy_mjc_tick_batch(int32_t nworld, const alloy_mjc_config* const* cfg, const mjModel* const* m,
                         mjData* const* d, const uint8_t* active, int32_t nstep, alloy_mjc_out* out,
                         int32_t* status);

/* The scoped noslip alone, on d after mj_fwdConstraint: writes efc_force, qfrc_constraint, qacc; counters in out
 * (may be NULL). */
int alloy_mjc_noslip(const alloy_mjc_config* cfg, const mjModel* m, mjData* d, alloy_mjc_out* out);

/* mjc_BoxBox on geoms g1, g2 at d's current poses (geom_xpos / geom_xmat), then the line rule when line = 1.
 * Writes up to mjMAXCONPAIR contacts as pos[3], normal[3], dist; returns the count (negative: bad arguments). */
int alloy_mjc_boxbox(const mjModel* m, mjData* d, int32_t g1, int32_t g2, double margin, int32_t line,
                     double* pos, double* normal, double* dist);

/* sizeof(alloy_mjc_config), sizeof(alloy_mjc_out): the ctypes mirror's layout check (abi.py) */
void alloy_mjc_struct_sizes(int32_t* sizes);

/* Puts MuJoCo's box-box routine back into mjCOLLISIONFUNC (tests); the next tick with line on re-installs. */
void alloy_mjc_unhook(void);

#endif  /* ALLOY_MJC_H_ */
