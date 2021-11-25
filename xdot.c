#include "xdot.h"

real xdot(const int n, const real *x, const int incx, const real *y, const int incy)
{

#if defined(SEPARATE_SPECIAL_CASE)

  if (incx == 1 && incy == 1) {
    
    real sum = ZERO;
    for (int i = 0; i < n; i ++) { // LOOP WAS VECTORIZED.
      sum += x[i] * y[i];
    }
    return sum;
    
  } else {

#endif
    
    //    int ix = 0;
    //    int iy = 0;
    const real *X = x;
    const real *Y = y;
    real sum = ZERO;
    for (int i = 0; i < n; i ++) { // LOOP WAS VECTORIZED.
      //      sum += x[ix] * y[iy];
      //      ix += *incx;
      //      iy += *incy;
      sum += (*X) * (*Y);
      X += incx;
      Y += incy;
    }
    return sum;

#if defined(SEPARATE_SPECIAL_CASE)
  }
#endif

}
