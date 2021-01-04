/* Calculates the complete elliptic integral of the first kind:
K(k) = F(k, \pi / 2) = \int_{0}^{\pi /2} \frac{d \theta}{\sqrt{1 - k^2
\sin^2{\theta}}} and second kind: E(k) = E(k , \pi / 2) = \int_{0}^{\pi /2}
\sqrt{1 - k^2 \sin^2{\theta}} d \theta using the GSL implementation:
https://github.com/ampl/gsl/blob/master/specfunc/ellint.c
*/

#pragma once

#define PREC_DOUBLE 2e-16
#define DBL_EPSILON 2.2204460492503131e-16
#define SQRT_DBL_EPSILON 1.4901161193847656e-08
#define DBL_MIN 2.2250738585072014e-308
#define DBL_MAX 1.7976931348623157e+308
#define ERROR_SELECT_2(a, b)                                                   \
  ((a) != SUCCESS ? (a) : ((b) != SUCCESS ? (b) : SUCCESS))

#include "Utilities/Definitions.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

namespace Math {

enum Status {
  SUCCESS = 0,
  FAILURE = -1,
  DOMAIN_ERROR = -2,
  MAX_ITERATION = -3
};

typedef struct {
  ActsScalar val;
  ActsScalar err;
} Result;

ActsScalar max3(const ActsScalar a, const ActsScalar b, const ActsScalar c) {
  ActsScalar d = std::max(a, b);
  return std::max(d, c);
}

ActsScalar min3(const ActsScalar a, const ActsScalar b, const ActsScalar c) {
  ActsScalar d = std::min(a, b);
  return std::min(d, c);
}

int ellint_RF_e(ActsScalar x, ActsScalar y, ActsScalar z, Result &result);
int ellint_Kcomp_e(ActsScalar k, Result &result);
int ellint_RD_e(ActsScalar x, ActsScalar y, ActsScalar z, Result &result);
int ellint_Ecomp_e(ActsScalar k, Result &result);

ActsScalar comp_ellint_1(ActsScalar k) {
  Result re;
  int status = ellint_Kcomp_e(k, re);
  if (status == SUCCESS) {
    return re.val;
  }
  printf("Failure status %d at line %d in %s.\n", status, __LINE__, __FILE__);
  return 0.;
}

ActsScalar comp_ellint_2(ActsScalar k) {
  Result re;
  int status = ellint_Ecomp_e(k, re);
  if (status == SUCCESS) {
    return re.val;
  }
  printf("Failure status %d at line %d in %s.\n", status, __LINE__, __FILE__);
  return 0.;
}

/* [Carlson, Numer. Math. 33 (1979) 1, (4.5)] */
int ellint_Kcomp_e(ActsScalar k, Result &result) {
  if (k * k >= 1.0) {
    return DOMAIN_ERROR;
  } else if (k * k >= 1.0 - SQRT_DBL_EPSILON) {
    /* [Abramowitz+Stegun, 17.3.34] */
    const ActsScalar y = 1.0 - k * k;
    const ActsScalar a[] = {1.38629436112, 0.09666344259, 0.03590092383};
    const ActsScalar b[] = {0.5, 0.12498593597, 0.06880248576};
    const ActsScalar ta = a[0] + y * (a[1] + y * a[2]);
    const ActsScalar tb = -log(y) * (b[0] + y * (b[1] + y * b[2]));
    result.val = ta + tb;
    result.err = 2.0 * DBL_EPSILON * (std::fabs(result.val) + std::fabs(k / y));
    return SUCCESS;
  } else {
    ActsScalar y = 1.0 - k * k;
    int status = ellint_RF_e(0.0, y, 1.0, result);
    result.err += 0.5 * DBL_EPSILON / y;
    return status;
  }
}

int ellint_RF_e(ActsScalar x, ActsScalar y, ActsScalar z, Result &result) {
  const ActsScalar lolim = 5.0 * DBL_MIN;
  const ActsScalar uplim = 0.2 * DBL_MAX;
  const ActsScalar errtol = 0.001;
  const ActsScalar prec = DBL_EPSILON;
  const int nmax = 10000;

  if (x < 0.0 || y < 0.0 || z < 0.0) {
    return DOMAIN_ERROR;
  } else if (x + y < lolim || x + z < lolim || y + z < lolim) {
    return DOMAIN_ERROR;
  } else if (max3(x, y, z) < uplim) {
    const ActsScalar c1 = 1.0 / 24.0;
    const ActsScalar c2 = 3.0 / 44.0;
    const ActsScalar c3 = 1.0 / 14.0;
    ActsScalar xn = x;
    ActsScalar yn = y;
    ActsScalar zn = z;
    ActsScalar mu, xndev, yndev, zndev, e2, e3, s;
    int n = 0;
    while (1) {
      ActsScalar epslon, lamda;
      ActsScalar xnroot, ynroot, znroot;
      mu = (xn + yn + zn) / 3.0;
      xndev = 2.0 - (mu + xn) / mu;
      yndev = 2.0 - (mu + yn) / mu;
      zndev = 2.0 - (mu + zn) / mu;
      epslon = max3(std::fabs(xndev), std::fabs(yndev), std::fabs(zndev));
      if (epslon < errtol)
        break;
      xnroot = std::sqrt(xn);
      ynroot = std::sqrt(yn);
      znroot = std::sqrt(zn);
      lamda = xnroot * (ynroot + znroot) + ynroot * znroot;
      xn = (xn + lamda) * 0.25;
      yn = (yn + lamda) * 0.25;
      zn = (zn + lamda) * 0.25;
      n++;
      if (n == nmax) {
        return MAX_ITERATION;
      }
    }
    e2 = xndev * yndev - zndev * zndev;
    e3 = xndev * yndev * zndev;
    s = 1.0 + (c1 * e2 - 0.1 - c2 * e3) * e2 + c3 * e3;
    result.val = s / std::sqrt(mu);
    result.err = prec * std::fabs(result.val);
    return SUCCESS;
  } else {
    return DOMAIN_ERROR;
  }
}

/* [Carlson, Numer. Math. 33 (1979) 1, (4.6)] */
int ellint_Ecomp_e(ActsScalar k, Result &result) {
  if (k * k >= 1.0) {
    return DOMAIN_ERROR;
  } else if (k * k >= 1.0 - SQRT_DBL_EPSILON) {
    /* [Abramowitz+Stegun, 17.3.36] */
    const ActsScalar y = 1.0 - k * k;
    const ActsScalar a[] = {0.44325141463, 0.06260601220, 0.04757383546};
    const ActsScalar b[] = {0.24998368310, 0.09200180037, 0.04069697526};
    const ActsScalar ta = 1.0 + y * (a[0] + y * (a[1] + a[2] * y));
    const ActsScalar tb = -y * std::log(y) * (b[0] + y * (b[1] + b[2] * y));
    result.val = ta + tb;
    result.err = 2.0 * DBL_EPSILON * result.val;
    return SUCCESS;
  } else {
    Result rf;
    Result rd;
    const ActsScalar y = 1.0 - k * k;
    const int rfstatus = ellint_RF_e(0.0, y, 1.0, rf);
    const int rdstatus = ellint_RD_e(0.0, y, 1.0, rd);
    result.val = rf.val - k * k / 3.0 * rd.val;
    result.err = rf.err + k * k / 3.0 * rd.err;
    return ERROR_SELECT_2(rfstatus, rdstatus);
  }
}

int ellint_RD_e(ActsScalar x, ActsScalar y, ActsScalar z, Result &result) {
  const ActsScalar errtol = 0.001;
  const ActsScalar prec = DBL_EPSILON;
  const ActsScalar lolim = 2.0 / pow(DBL_MAX, 2.0 / 3.0);
  const ActsScalar uplim = pow(0.1 * errtol / DBL_MIN, 2.0 / 3.0);
  const int nmax = 10000;

  if (std::min(x, y) < 0.0 || std::min(x + y, z) < lolim) {
    return DOMAIN_ERROR;
  } else if (max3(x, y, z) < uplim) {
    const ActsScalar c1 = 3.0 / 14.0;
    const ActsScalar c2 = 1.0 / 6.0;
    const ActsScalar c3 = 9.0 / 22.0;
    const ActsScalar c4 = 3.0 / 26.0;
    ActsScalar xn = x;
    ActsScalar yn = y;
    ActsScalar zn = z;
    ActsScalar sigma = 0.0;
    ActsScalar power4 = 1.0;
    ActsScalar ea, eb, ec, ed, ef, s1, s2;
    ActsScalar mu, xndev, yndev, zndev;
    int n = 0;
    while (1) {
      ActsScalar xnroot, ynroot, znroot, lamda;
      ActsScalar epslon;
      mu = (xn + yn + 3.0 * zn) * 0.2;
      xndev = (mu - xn) / mu;
      yndev = (mu - yn) / mu;
      zndev = (mu - zn) / mu;
      epslon = max3(std::fabs(xndev), std::fabs(yndev), std::fabs(zndev));
      if (epslon < errtol)
        break;
      xnroot = std::sqrt(xn);
      ynroot = std::sqrt(yn);
      znroot = std::sqrt(zn);
      lamda = xnroot * (ynroot + znroot) + ynroot * znroot;
      sigma += power4 / (znroot * (zn + lamda));
      power4 *= 0.25;
      xn = (xn + lamda) * 0.25;
      yn = (yn + lamda) * 0.25;
      zn = (zn + lamda) * 0.25;
      n++;
      if (n == nmax) {
        return MAX_ITERATION;
      }
    }
    ea = xndev * yndev;
    eb = zndev * zndev;
    ec = ea - eb;
    ed = ea - 6.0 * eb;
    ef = ed + ec + ec;
    s1 = ed * (-c1 + 0.25 * c3 * ed - 1.5 * c4 * zndev * ef);
    s2 = zndev * (c2 * ef + zndev * (-c3 * ec + zndev * c4 * ea));
    result.val = 3.0 * sigma + power4 * (1.0 + s1 + s2) / (mu * std::sqrt(mu));
    result.err = prec * std::fabs(result.val);
    return SUCCESS;
  } else {
    return DOMAIN_ERROR;
  }
}

} // namespace Math
