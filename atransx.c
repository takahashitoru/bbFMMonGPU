#include "atransx.h"

// A  : m by n, (column-major)
// A^T: n by m, row-major
// x  : m
// y  : n
// Compute y += A^T * x

void atransx(const int m, const int n, const real *atrans, const real *x, real *y)
{
#if(0) // unrolling; slower
  const int m2 = m / 2;
  const int mod = m % 2;

  for (int i = 0; i < n; i ++) {
    real sum = ZERO;
    const real *atransi = &(atrans[m * i]); // j=0
    int j = 0;
    for (int j2 = 0; j2 < m2; j2 ++) { // LOOP WAS VECTORIZED.
      sum += atransi[j] * x[j];
      j ++;
      sum += atransi[j] * x[j];
      j ++;
    }
    if (mod) {
      sum += atransi[j] * x[j];
    }
    y[i] += sum;
  }

#else

  for (int i = 0; i < n; i ++) {
    real sum = ZERO;
    const real *atransi = &(atrans[m * i]); // j=0
    //    const real *xj = x; // j=0
    //    const real *atransij = &(atrans[m * i]); // j=0
    for (int j = 0; j < m; j ++) { // LOOP WAS VECTORIZED.
      sum += atransi[j] * x[j];
      //      sum += (*atransij) * (*xj);
      //      atransij ++;
      //      xj ++;
    }
    y[i] += sum;
  }

#endif

}
