/* Internal declarations shared by the extension's C files (not part of the ABI). */
#ifndef ALLOY_MJC_INTERNAL_H_
#define ALLOY_MJC_INTERNAL_H_

#include "alloy_mjc.h"

#define MJC_MAXNF 5  /* friction rows of the largest contact (condim 6) */

/* scoped noslip on d after mj_fwdConstraint (noslip.c); cfg already checked */
int mjc_noslip_run(const alloy_mjc_config* cfg, const mjModel* m, mjData* d, alloy_mjc_out* out);

/* friction-row projections (cone.c), as alloy.contact.noslip_cone */
double mjc_dof_force(double adiag, double frictionloss, double jq_minus_aref, double old, double* change);
/* one elliptic contact's nf friction rows with the normal force fixed: Ac (nf x nf, row-major, diagonal floored
 * at 1e-10 by the caller), res = J qacc - aref, old forces; writes the new forces, returns the cost change and the
 * condim-6 QCQP multiplier in *la (warm start la0) */
double mjc_contact_block(const double* Ac, const double* res, const double* old, const double* friction,
                         double normal, int nf, double la0, double* force, double* la);

/* box-box line hook (boxbox_line.c): active on this thread between begin and end */
void mjc_line_begin(const alloy_mjc_config* cfg, alloy_mjc_out* out);
void mjc_line_end(void);
/* the line rule on one single-contact box-box result: 2 when con[0..1] were rewritten, else 0 */
int mjc_line_apply(const mjModel* m, const mjData* d, int g1, int g2, mjtNum margin, mjPreContact* con);

#endif  /* ALLOY_MJC_INTERNAL_H_ */
