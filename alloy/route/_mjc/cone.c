/* Friction-row projections of the scoped noslip, as alloy.contact.noslip_cone (MuJoCo's noslip: dry-friction clamp,
 * elliptic-cone QCQP with the normal force fixed), in float64.
 *
 * qcqp2 / qcqp3 are MuJoCo's mju_QCQP2 / mju_QCQP3 (by way of MuJoCo Warp's _qcqp2 / _qcqp3, same operations).
 * qcqpn_warm is the generic QCQP's Newton iteration on the multiplier la, started at la0 (alloy's qcqp5_warm; la0 = 0
 * is MuJoCo's start). Alloy's two differences from MuJoCo's noslip are kept (noslip_cone docstring): a result outside
 * the cone is put on it even when the QCQP reports "inactive", and a non-finite result becomes zero friction.
 */
#include <math.h>
#include <string.h>

#include "mjc_internal.h"

#define QCQP_ITERS 20
#define QCQP_EPS 1.0e-10

double mjc_dof_force(double adiag, double frictionloss, double r, double old, double* change) {
  double force = old - r / adiag;            /* r = J qacc - aref; adiag floored at mjMINVAL by the caller */
  if (force < -frictionloss) {
    force = -frictionloss;
  } else if (force > frictionloss) {
    force = frictionloss;
  }
  double delta = force - old;
  *change = 0.5 * delta * delta * adiag + delta * r;
  return force;
}

static int qcqp2(const double* Ain, const double* bin, const double* d, double r, double* x) {
  double b1 = bin[0] * d[0], b2 = bin[1] * d[1];
  double A11 = Ain[0] * d[0] * d[0], A22 = Ain[3] * d[1] * d[1], A12 = Ain[1] * d[0] * d[1];
  double la = 0.0, v1 = 0.0, v2 = 0.0;
  for (int it = 0; it < QCQP_ITERS; it++) {
    double det = (A11 + la) * (A22 + la) - A12 * A12;
    if (det < QCQP_EPS) {
      x[0] = x[1] = 0.0;
      return 0;
    }
    double detinv = 1.0 / det;
    double P11 = (A22 + la) * detinv, P22 = (A11 + la) * detinv, P12 = -A12 * detinv;
    v1 = -P11 * b1 - P12 * b2;
    v2 = -P12 * b1 - P22 * b2;
    double val = v1 * v1 + v2 * v2 - r * r;
    if (val < QCQP_EPS) {
      break;
    }
    double deriv = -2.0 * (P11 * v1 * v1 + 2.0 * P12 * v1 * v2 + P22 * v2 * v2);
    double delta = -val / deriv;
    if (delta < QCQP_EPS) {
      break;
    }
    la += delta;
  }
  x[0] = v1 * d[0];
  x[1] = v2 * d[1];
  return la != 0.0;
}

static int qcqp3(const double* Ain, const double* bin, const double* d, double r, double* x) {
  double b1 = bin[0] * d[0], b2 = bin[1] * d[1], b3 = bin[2] * d[2];
  double A11 = Ain[0] * d[0] * d[0], A22 = Ain[4] * d[1] * d[1], A33 = Ain[8] * d[2] * d[2];
  double A12 = Ain[1] * d[0] * d[1], A13 = Ain[2] * d[0] * d[2], A23 = Ain[5] * d[1] * d[2];
  double la = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0;
  for (int it = 0; it < QCQP_ITERS; it++) {
    double P11 = (A22 + la) * (A33 + la) - A23 * A23;
    double P22 = (A11 + la) * (A33 + la) - A13 * A13;
    double P33 = (A11 + la) * (A22 + la) - A12 * A12;
    double P12 = A13 * A23 - A12 * (A33 + la);
    double P13 = A12 * A23 - A13 * (A22 + la);
    double P23 = A12 * A13 - A23 * (A11 + la);
    double det = (A11 + la) * P11 + A12 * P12 + A13 * P13;
    if (det < QCQP_EPS) {
      x[0] = x[1] = x[2] = 0.0;
      return 0;
    }
    double detinv = 1.0 / det;
    P11 *= detinv;
    P22 *= detinv;
    P33 *= detinv;
    P12 *= detinv;
    P13 *= detinv;
    P23 *= detinv;
    v1 = -P11 * b1 - P12 * b2 - P13 * b3;
    v2 = -P12 * b1 - P22 * b2 - P23 * b3;
    v3 = -P13 * b1 - P23 * b2 - P33 * b3;
    double val = v1 * v1 + v2 * v2 + v3 * v3 - r * r;
    if (val < QCQP_EPS) {
      break;
    }
    double deriv = -2.0 * (P11 * v1 * v1 + P22 * v2 * v2 + P33 * v3 * v3) -
                   4.0 * (P12 * v1 * v2 + P13 * v1 * v3 + P23 * v2 * v3);
    double delta = -val / deriv;
    if (delta < QCQP_EPS) {
      break;
    }
    la += delta;
  }
  x[0] = v1 * d[0];
  x[1] = v2 * d[1];
  x[2] = v3 * d[2];
  return la != 0.0;
}

/* lower Cholesky factor of A + la I (n x n, row-major), MuJoCo's 1e-10 rank threshold; 0 if not SPD */
static int chol_shift(const double* A, int n, double la, double* L) {
  memcpy(L, A, sizeof(double) * n * n);
  for (int i = 0; i < n; i++) {
    L[i * n + i] = A[i * n + i] + la;
  }
  for (int j = 0; j < n; j++) {
    double diag = L[j * n + j];
    for (int k = 0; k < j; k++) {
      diag -= L[j * n + k] * L[j * n + k];
    }
    if (diag < QCQP_EPS) {
      return 0;
    }
    diag = sqrt(diag);
    L[j * n + j] = diag;
    for (int i = j + 1; i < n; i++) {
      double val = L[i * n + j];
      for (int k = 0; k < j; k++) {
        val -= L[i * n + k] * L[j * n + k];
      }
      L[i * n + j] = val / diag;
    }
  }
  return 1;
}

static void chol_solve(const double* L, int n, const double* b, double* x) {
  for (int i = 0; i < n; i++) {
    double val = b[i];
    for (int k = 0; k < i; k++) {
      val -= L[i * n + k] * x[k];
    }
    x[i] = val / L[i * n + i];
  }
  for (int i = n - 1; i >= 0; i--) {
    double val = x[i];
    for (int k = i + 1; k < n; k++) {
      val -= L[k * n + i] * x[k];
    }
    x[i] = val / L[i * n + i];
  }
}

static double dotn(const double* a, const double* b, int n) {
  double s = 0.0;
  for (int i = 0; i < n; i++) {
    s += a[i] * b[i];
  }
  return s;
}

/* Newton on phi(la) = |x(la)|^2 - r^2 from la0 (alloy.contact.noslip_cone.qcqp5_warm, any n <= 5) */
static int qcqpn_warm(int n, const double* Ain, const double* bin, const double* d, double r, double la0, double* x,
                      double* la_out) {
  double A[MJC_MAXNF * MJC_MAXNF], b[MJC_MAXNF], L[MJC_MAXNF * MJC_MAXNF], res[MJC_MAXNF], t[MJC_MAXNF];
  for (int i = 0; i < n; i++) {
    b[i] = bin[i] * d[i];
    for (int j = 0; j < n; j++) {
      A[i * n + j] = Ain[i * n + j] * d[i] * d[j];
    }
  }
  double la = la0;
  for (int it = 0; it < QCQP_ITERS; it++) {
    if (!chol_shift(A, n, la, L)) {
      memset(x, 0, sizeof(double) * n);
      *la_out = 0.0;
      return 0;
    }
    chol_solve(L, n, b, res);
    for (int i = 0; i < n; i++) {
      res[i] = -res[i];
    }
    double val = dotn(res, res, n) - r * r;
    if (val < QCQP_EPS) {
      if (la == la0 && la0 > 0.0 && val < 0.0) {   /* warm start right of the root: one tangent step left */
        chol_solve(L, n, res, t);
        la = fmax(la + val / (2.0 * dotn(res, t, n)), 0.0);
        continue;
      }
      break;
    }
    chol_solve(L, n, res, t);
    double deriv = -2.0 * dotn(res, t, n);
    double delta = -val / deriv;
    if (delta < QCQP_EPS) {
      break;
    }
    la += delta;
  }
  for (int i = 0; i < n; i++) {
    x[i] = res[i] * d[i];
  }
  *la_out = la;
  return la != 0.0;
}

/* friction force minimising 0.5 f'Af + f'b in the elliptic cone of the fixed normal force; returns the condim-6
 * multiplier (0 for the other sizes) */
static double cone_force(const double* Ac, const double* bc, const double* mu, double normal, int nf, double la0,
                         double* force) {
  memset(force, 0, sizeof(double) * nf);
  if (normal < mjMINVAL) {
    return 0.0;
  }
  int active = 0;
  double la = 0.0, unused = 0.0;
  if (nf == 2) {
    active = qcqp2(Ac, bc, mu, normal, force);
  } else if (nf == 3) {
    active = qcqp3(Ac, bc, mu, normal, force);
  } else if (nf == 5) {
    active = qcqpn_warm(5, Ac, bc, mu, normal, la0, force, &la);
  } else {
    active = qcqpn_warm(nf, Ac, bc, mu, normal, 0.0, force, &unused);
  }
  double ss = 0.0;
  for (int k = 0; k < nf; k++) {
    double fk = force[k] / (mu[k] != 0.0 ? mu[k] : mjMINVAL);
    ss += fk * fk;
  }
  if (!isfinite(ss)) {
    memset(force, 0, sizeof(double) * nf);
    return 0.0;
  }
  if (active || ss > normal * normal) {   /* on or outside the cone: put the force on the ellipsoid */
    double scl = sqrt(normal * normal / fmax(mjMINVAL, ss));
    for (int k = 0; k < nf; k++) {
      force[k] *= scl;
    }
  }
  return la;
}

double mjc_contact_block(const double* Ac, const double* res, const double* old, const double* friction,
                         double normal, int nf, double la0, double* force, double* la) {
  double bc[MJC_MAXNF];
  for (int k = 0; k < nf; k++) {
    bc[k] = res[k];
    for (int l = 0; l < nf; l++) {
      bc[k] -= Ac[k * nf + l] * old[l];
    }
  }
  *la = cone_force(Ac, bc, friction, normal, nf, la0, force);
  double change = 0.0;
  for (int k = 0; k < nf; k++) {
    double ad = 0.0;
    for (int l = 0; l < nf; l++) {
      ad += Ac[k * nf + l] * (force[l] - old[l]);
    }
    double dk = force[k] - old[k];
    change += 0.5 * dk * ad + dk * res[k];
  }
  if (change > 1.0e-10) {   /* a cost increase keeps the old forces (MuJoCo's costChange) */
    memcpy(force, old, sizeof(double) * nf);
    return 0.0;
  }
  return change;
}
