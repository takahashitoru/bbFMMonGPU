#include "m2l_aux.h"

void m2l_aux_comp_Ktable(int *Ktable)
{
  /* (input/output) Ktable: Integer array of dimension 343 */

  /* Initialize lookup table */
  for (int i = 0; i < 343; i ++) {
    Ktable[i] = NULL_KINDEX;
  }

  /* Create lookup table */
  int ncell = 0;
  int ninteract = 0;
  for (int vx = - 3; vx < 4; vx ++) {
    for (int vy = - 3; vy < 4; vy ++) {
      for (int vz = - 3; vz < 4; vz ++) {
	if (abs(vx) > 1 || abs(vy) > 1 || abs(vz) > 1) {
	  Ktable[ncell] = ninteract;
	  ninteract ++;
	}
	ncell ++;
      }
    }
  }
}	


void m2l_aux_convert_K_to_Kanother_for_cluster_blocking(int r, int P, int Q, real *K, real *Kanother)
{
  /* (input) r: Dimension of K-matrices.  P: Number of tiles for
     row. Q: Number of tiles for column.  K: real-type array of size
     316*r^2. This contains 316 K-matrices. Each matrix is stored in
     column-major.  (output) Kanother: real-type array of size
     316*r^2. This contains 316 K-matrices. Each matrix is stored in
     double column-major. */

  int nrow = r / P; // Number of rows per tile
  int ncol = r / Q; // Number of columns per tile
  
  for (int k = 0; k < 316; k ++) {
    int count = 0;
    for (int q = 0; q < Q; q ++) {
      for (int p = 0; p < P; p ++) {
	for (int col = 0; col < ncol; col ++) {
	  int j = q * ncol + col;
	  for (int row = 0; row < nrow; row ++) {
	    int i = p * nrow + row;
	    Kanother[r * r * k + count] = K[(k * r + j) * r + i];
	    count ++;
	  }
	}
      }
    }
  }
}


void m2l_aux_convert_K_to_Kanother_for_ij_blocking(int r, int *Ktable, real *K, real4 *Kanother)
{
  /* (input) r: Dimension of K-matrices.  Ktable: Integer array of
     dimension 343. This is created by m2l_aux_comp_Ktable.  K:
     real-type array of dimension 316*r^2. This contains 316
     K-matrices. Each matrix is stored in column-major.  (output)
     Kanother: 4D-vector-type array of dimension 316*(r/4)*r. This
     contains 316 K-matrices. Each matrix is stored in a special
     format for ij-blocking scheme. */

  /* Assertion */
  ASSERT(r % 4 == 0);

  /* Initialize index for interacting cells */
  int n = 0;

  /* Loops over M2L vectors. The order of loops is the reverse to
     those in creating Ktable. Never change the order. */
  for (int vz = - 3; vz < 4; vz ++) {
    for (int vy = - 3; vy < 4; vy ++) {
      for (int vx = - 3; vx < 4; vx ++) {
	int k = Ktable[49 * (vx + 3) + 7 * (vy + 3) + (vz + 3)]; // k=[0:316) or NULL_KINDEX
	if (k != NULL_KINDEX) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	  /* Loop over columns */
	  for (int j = 0; j < r; j ++) {  // no unrolling
	    /* Loop over rows */
	    for (int i = 0; i < r; i += 4) { // unrolling 4x
	      int inew = (j * (r / 4) + (i / 4)) * 316 + n;
	      int iold = i + r * (j + r * k);
	      Kanother[inew].x = K[iold    ]; // row i
	      Kanother[inew].y = K[iold + 1]; // row i+1
	      Kanother[inew].z = K[iold + 2]; // row i+2
	      Kanother[inew].w = K[iold + 3]; // row i+3
	    }
	  }
	  n ++;
	}
      }
    }
  }
}


void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row4_col1(int r, int *Ktable, real *K, real4 *Kanother)
{
  m2l_aux_convert_K_to_Kanother_for_ij_blocking(r, Ktable, K, Kanother);
}


void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row1_col2(int r, int *Ktable, real *K, real2 *Kanother)
{
  /* (input) r: Dimension of K-matrices.  Ktable: Integer array of
     dimension 343. This is created by m2l_aux_comp_Ktable.  K:
     real-type array of dimension 316*r^2. This contains 316
     K-matrices. Each matrix is stored in column-major.  (output)
     Kanother2: 2D-vector-type array of dimension 316*r*(r/2). This
     contains 316 K-matrices. Each matrix is stored in a special
     format for ij-blocking scheme. */

  /* Assertion */
  ASSERT(r % 2 == 0);

  /* Initialize index for interacting cells */
  int n = 0;

  /* Loops over M2L vectors. The order of loops is the reverse to
     those in creating Ktable. Never change the order. */
  for (int vz = - 3; vz < 4; vz ++) {
    for (int vy = - 3; vy < 4; vy ++) {
      for (int vx = - 3; vx < 4; vx ++) {
	int k = Ktable[49 * (vx + 3) + 7 * (vy + 3) + (vz + 3)]; // k=[0:316) or NULL_KINDEX
	if (k != NULL_KINDEX) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	  /* Loop over columns */
	  for (int j = 0; j < r; j += 2) { // unrolling 2x
	    /* Loop over rows */
	    for (int i = 0; i < r; i ++) { // no unrolling
	      int inew = ((j / 2) * r + i) * 316 + n;
	      int iold = i + r * (j + r * k);
	      Kanother[inew].x = K[iold    ]; // row i, column j
	      Kanother[inew].y = K[iold + r]; // row i, column j+1
	    }
	  }
	  n ++;
	}
      }
    }
  }
}


void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row8_col1(int r, int *Ktable, real *K, real8 *Kanother)
{
  /* (input) r: Dimension of K-matrices.  Ktable: Integer array of
     dimension 343. This is created by m2l_aux_comp_Ktable.  K:
     real-type array of dimension 316*r^2. This contains 316
     K-matrices. Each matrix is stored in column-major.  (output)
     Kanother: 8D-vector-type array of dimension 316*(r/8)*r. This
     contains 316 K-matrices. Each matrix is stored in a special
     format for ij-blocking scheme. */

  /* Assertion */
  ASSERT(r % 8 == 0);

  /* Initialize index for interacting cells */
  int n = 0;

  /* Loops over M2L vectors. The order of loops is the reverse to
     those in creating Ktable. Never change the order. */
  for (int vz = - 3; vz < 4; vz ++) {
    for (int vy = - 3; vy < 4; vy ++) {
      for (int vx = - 3; vx < 4; vx ++) {
	int k = Ktable[49 * (vx + 3) + 7 * (vy + 3) + (vz + 3)]; // k=[0:316) or NULL_KINDEX
	if (k != NULL_KINDEX) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	  /* Loop over columns */
	  for (int j = 0; j < r; j ++) { // no unrolling
	    /* Loop over rows */
	    for (int i = 0; i < r; i += 8) { // unrolling 8x
	      int inew = (j * (r / 8) + (i / 8)) * 316 + n;
	      int iold = i + r * (j + r * k);
	      Kanother[inew].a = K[iold    ]; // row i
	      Kanother[inew].b = K[iold + 1]; // row i+1
	      Kanother[inew].c = K[iold + 2]; // row i+2
	      Kanother[inew].d = K[iold + 3]; // row i+3
	      Kanother[inew].e = K[iold + 4]; // row i+4
	      Kanother[inew].f = K[iold + 5]; // row i+5
	      Kanother[inew].g = K[iold + 6]; // row i+6
	      Kanother[inew].h = K[iold + 7]; // row i+7
	    }
	  }
	  n ++;
	}
      }
    }
  }
}


void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row16_col1(int r, int *Ktable, real *K, real16 *Kanother)
{
  /* (input) r: Dimension of K-matrices.  Ktable: Integer array of
     dimension 343. This is created by m2l_aux_comp_Ktable.  K:
     real-type array of dimension 316*r^2. This contains 316
     K-matrices. Each matrix is stored in column-major.  (output)
     Kanother: 16D-vector-type array of dimension 316*(r/16)*r. This
     contains 316 K-matrices. Each matrix is stored in a special
     format for ij-blocking scheme. */

  /* Assertion */
  ASSERT(r % 16 == 0);

  /* Initialize index for interacting cells */
  int n = 0;

  /* Loops over M2L vectors. The order of loops is the reverse to
     those in creating Ktable. Never change the order. */
  for (int vz = - 3; vz < 4; vz ++) {
    for (int vy = - 3; vy < 4; vy ++) {
      for (int vx = - 3; vx < 4; vx ++) {
	int k = Ktable[49 * (vx + 3) + 7 * (vy + 3) + (vz + 3)]; // k=[0:316) or NULL_KINDEX
	if (k != NULL_KINDEX) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	  /* Loop over columns */
	  for (int j = 0; j < r; j ++) { // no unrolling
	    /* Loop over rows */
	    for (int i = 0; i < r; i += 16) { // unrolling 16x
	      int inew = (j * (r / 16) + (i / 16)) * 316 + n;
	      int iold = i + r * (j + r * k);
	      Kanother[inew].a = K[iold     ]; // row i
	      Kanother[inew].b = K[iold +  1]; // row i+1
	      Kanother[inew].c = K[iold +  2]; // row i+2
	      Kanother[inew].d = K[iold +  3]; // row i+3
	      Kanother[inew].e = K[iold +  4]; // row i+4
	      Kanother[inew].f = K[iold +  5]; // row i+5
	      Kanother[inew].g = K[iold +  6]; // row i+6
	      Kanother[inew].h = K[iold +  7]; // row i+7
	      Kanother[inew].i = K[iold +  8]; // row i+8
	      Kanother[inew].j = K[iold +  9]; // row i+9
	      Kanother[inew].k = K[iold + 10]; // row i+10
	      Kanother[inew].l = K[iold + 11]; // row i+11
	      Kanother[inew].m = K[iold + 12]; // row i+12
	      Kanother[inew].n = K[iold + 13]; // row i+13
	      Kanother[inew].o = K[iold + 14]; // row i+14
	      Kanother[inew].p = K[iold + 15]; // row i+15
	    }
	  }
	  n ++;
	}
      }
    }
  }
}


void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row8_col2(int r, int *Ktable, real *K, real8x2 *Kanother)
{
  /* (input) r: Dimension of K-matrices.  Ktable: Integer array of
     dimension 343. This is created by m2l_aux_comp_Ktable.  K:
     real-type array of dimension 316*r^2. This contains 316
     K-matrices. Each matrix is stored in column-major.  (output)
     Kanother: 8x2D-vector-type array of dimension
     316*(r/8)*(r/2). This contains 316 K-matrices. Each matrix is
     stored in a special format for ij-blocking scheme. */

  /* Assertion */
  ASSERT(r % 8 == 0);

  /* Initialize index for interacting cells */
  int n = 0;

  /* Loops over M2L vectors. The order of loops is the reverse to
     those in creating Ktable. Never change the order. */
  for (int vz = - 3; vz < 4; vz ++) {
    for (int vy = - 3; vy < 4; vy ++) {
      for (int vx = - 3; vx < 4; vx ++) {
	int k = Ktable[49 * (vx + 3) + 7 * (vy + 3) + (vz + 3)]; // k=[0:316) or NULL_KINDEX
	if (k != NULL_KINDEX) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	  /* Loop over columns */
	  for (int j = 0; j < r; j += 2) { // unrolling 2x
	    /* Loop over rows */
	    for (int i = 0; i < r; i += 8) { // unrolling 8x
	      int inew = ((j / 2) * (r / 8) + (i / 8)) * 316 + n;
	      int iold = i + r * (j + r * k);
	      Kanother[inew].aa = K[iold    ]; // row i  , col j
	      Kanother[inew].ba = K[iold + 1]; // row i+1, col j
	      Kanother[inew].ca = K[iold + 2]; // row i+2, col j
	      Kanother[inew].da = K[iold + 3]; // row i+3, col j
	      Kanother[inew].ea = K[iold + 4]; // row i+4, col j
	      Kanother[inew].fa = K[iold + 5]; // row i+5, col j
	      Kanother[inew].ga = K[iold + 6]; // row i+6, col j
	      Kanother[inew].ha = K[iold + 7]; // row i+7, col j
	      
	      Kanother[inew].ab = K[iold     + r]; // row i  , col j+1
	      Kanother[inew].bb = K[iold + 1 + r]; // row i+1, col j+1
	      Kanother[inew].cb = K[iold + 2 + r]; // row i+2, col j+1
	      Kanother[inew].db = K[iold + 3 + r]; // row i+3, col j+1
	      Kanother[inew].eb = K[iold + 4 + r]; // row i+4, col j+1
	      Kanother[inew].fb = K[iold + 5 + r]; // row i+5, col j+1
	      Kanother[inew].gb = K[iold + 6 + r]; // row i+6, col j+1
	      Kanother[inew].hb = K[iold + 7 + r]; // row i+7, col j+1
	    }
	  }
	  n ++;
	}
      }
    }
  }
}


void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row4_col4(int r, int *Ktable, real *K, real4x4 *Kanother)
{
  /* (input) r: Dimension of K-matrices.  Ktable: Integer array of
     dimension 343. This is created by m2l_aux_comp_Ktable.  K:
     real-type array of dimension 316*r^2. This contains 316
     K-matrices. Each matrix is stored in column-major.  (output)
     Kanother: 4x4D-vector-type array of dimension
     316*(r/4)*(r/4). This contains 316 K-matrices. Each matrix is
     stored in a special format for ij-blocking scheme. */

  /* Assertion */
  ASSERT(r % 4 == 0);

  /* Initialize index for interacting cells */
  int n = 0;

  /* Loops over M2L vectors. The order of loops is the reverse to
     those in creating Ktable. Never change the order. */
  for (int vz = - 3; vz < 4; vz ++) {
    for (int vy = - 3; vy < 4; vy ++) {
      for (int vx = - 3; vx < 4; vx ++) {
	int k = Ktable[49 * (vx + 3) + 7 * (vy + 3) + (vz + 3)]; // k=[0:316) or NULL_KINDEX
	if (k != NULL_KINDEX) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	  /* Loop over columns */
	  for (int j = 0; j < r; j += 4) { // unrolling 4x
	    /* Loop over rows */
	    for (int i = 0; i < r; i += 4) { // unrolling 4x
	      int inew = ((j / 4) * (r / 4) + (i / 4)) * 316 + n;
	      int iold = i + r * (j + r * k);
	      Kanother[inew].xx = K[iold    ]; // row i  , col j
	      Kanother[inew].yx = K[iold + 1]; // row i+1, col j
	      Kanother[inew].zx = K[iold + 2]; // row i+2, col j
	      Kanother[inew].wx = K[iold + 3]; // row i+3, col j
	      iold += r;
	      Kanother[inew].xy = K[iold    ]; // row i  , col j+1
	      Kanother[inew].yy = K[iold + 1]; // row i+1, col j+1
	      Kanother[inew].zy = K[iold + 2]; // row i+2, col j+1
	      Kanother[inew].wy = K[iold + 3]; // row i+3, col j+1
	      iold += r;
	      Kanother[inew].xz = K[iold    ]; // row i  , col j+2
	      Kanother[inew].yz = K[iold + 1]; // row i+1, col j+2
	      Kanother[inew].zz = K[iold + 2]; // row i+2, col j+2
	      Kanother[inew].wz = K[iold + 3]; // row i+3, col j+2
	      iold += r;
	      Kanother[inew].xw = K[iold    ]; // row i  , col j+3
	      Kanother[inew].yw = K[iold + 1]; // row i+1, col j+3
	      Kanother[inew].zw = K[iold + 2]; // row i+2, col j+3
	      Kanother[inew].ww = K[iold + 3]; // row i+3, col j+3
	    }
	  }
	  n ++;
	}
      }
    }
  }
}


void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row4_col2(int r, int *Ktable, real *K, real4x2 *Kanother)
{
  /* (input) r: Dimension of K-matrices.  Ktable: Integer array of
     dimension 343. This is created by m2l_aux_comp_Ktable.  K:
     real-type array of dimension 316*r^2. This contains 316
     K-matrices. Each matrix is stored in column-major.  (output)
     Kanother: 4x2D-vector-type array of dimension
     316*(r/4)*(r/2). This contains 316 K-matrices. Each matrix is
     stored in a special format for ij-blocking scheme. */

  /* Assertion */
  ASSERT(r % 4 == 0);

  /* Initialize index for interacting cells */
  int n = 0;

  /* Loops over M2L vectors. The order of loops is the reverse to
     those in creating Ktable. Never change the order. */
  for (int vz = - 3; vz < 4; vz ++) {
    for (int vy = - 3; vy < 4; vy ++) {
      for (int vx = - 3; vx < 4; vx ++) {
	int k = Ktable[49 * (vx + 3) + 7 * (vy + 3) + (vz + 3)]; // k=[0:316) or NULL_KINDEX
	if (k != NULL_KINDEX) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	  /* Loop over columns */
	  for (int j = 0; j < r; j += 2) { // unrolling 2x
	    /* Loop over rows */
	    for (int i = 0; i < r; i += 4) { // unrolling 4x
	      int inew = ((j / 2) * (r / 4) + (i / 4)) * 316 + n;
	      int iold = i + r * (j + r * k);
	      Kanother[inew].xx = K[iold    ]; // row i  , col j
	      Kanother[inew].yx = K[iold + 1]; // row i+1, col j
	      Kanother[inew].zx = K[iold + 2]; // row i+2, col j
	      Kanother[inew].wx = K[iold + 3]; // row i+3, col j
	      iold += r;
	      Kanother[inew].xy = K[iold    ]; // row i  , col j+1
	      Kanother[inew].yy = K[iold + 1]; // row i+1, col j+1
	      Kanother[inew].zy = K[iold + 2]; // row i+2, col j+1
	      Kanother[inew].wy = K[iold + 3]; // row i+3, col j+1
	    }
	  }
	  n ++;
	}
      }
    }
  }
}


void m2l_aux_convert_K_to_Kanother_for_ij_blocking_row1_col1(int r, int *Ktable, real *K, real *Kanother)
{
  /* (input) r: Dimension of K-matrices.  Ktable: Integer array of
     dimension 343. This is created by m2l_aux_comp_Ktable.  K:
     real-type array of dimension 316*r^2. This contains 316
     K-matrices. Each matrix is stored in column-major.  (output)
     Kanother: real-type array of dimension 316*r*r. This contains 316
     K-matrices. Each matrix is stored in a special format for
     ij-blocking scheme. */

  /* Initialize index for interacting cells */
  int n = 0;

  /* Loops over M2L vectors. The order of loops is the reverse to
     those in creating Ktable. Never change the order. */
  for (int vz = - 3; vz < 4; vz ++) {
    for (int vy = - 3; vy < 4; vy ++) {
      for (int vx = - 3; vx < 4; vx ++) {
	int k = Ktable[49 * (vx + 3) + 7 * (vy + 3) + (vz + 3)]; // k=[0:316) or NULL_KINDEX
	if (k != NULL_KINDEX) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	  /* Loop over columns */
	  for (int j = 0; j < r; j ++) {  // no unrolling
	    /* Loop over rows */
	    for (int i = 0; i < r; i ++) { // no unrolling
	      int inew = (j * r + i) * 316 + n;
	      int iold = i + r * (j + r * k);
	      Kanother[inew] = K[iold];
	    }
	  }
	  n ++;
	}
      }
    }
  }
}


int m2l_aux_get_number_of_real_and_ghost_cells_for_ij_blocking(int level_start, int level_end)
{
  /* This function returns the number of both real and ghost cells in
     all the levels where the ij-blocking scheme is applied. */

  int n = 0;
  for (int i = level_start; i <= level_end; i ++) {
    int ncpe = POW2(i) + 4; // 2^i real cells and 4 ghost cells in level i
    n += ncpe * ncpe * ncpe;
  }
  return n;
}


int m2l_aux_get_starting_index_of_Manother_for_ij_blocking(int r, int level_start, int level)
{
  /* This function returns the starting index of Manother for the
     given level. */

  int s = 0;
  for (int i = level_start; i < level; i ++) {
    int ncpe = POW2(i) + 4; // 2^i real cells and 4 ghost cells in level i
    s += ncpe * ncpe * ncpe;
  }
  s *= r; // each cell has r elements
  return s;
}


void m2l_aux_convert_M_to_Manother_for_ij_blocking(int r, int level_start, int level_end, real3 *center, real L0, real *M, real *Manother)
{
  /* (input) r: Dimension of M-vectors. level_start: First level where
     the ij-blocking scheme is applied. level_end: Last level where
     the ij-blocking scheme is applied. center: 3D-vector-type array of
     size ncell (ncell stands for the number of all the cells in all
     the levels). center[i].{x,y,z} denote the central coordinates of
     the i-th cell. L0: Length of the root cell (0th cell).  M:
     real-type array of size r*ncell. This contains M-vectors for all
     the cells in the standard format. (output) Manother: real-type
     array of size r*\sum_{l=level_start}^{level_end}(2^l+4)^3. This
     must be initilized beforehand. This will contain M-vectors for
     all the real and ghost cells in the special format for
     ij-blocking scheme. */

  /* Loop over levels to which ij-blocking scheme is applied */
  for (int level = level_start; level <= level_end; level ++) {

    /* Number of real and ghost cells per edge */
    int ncpe = POW2(level) + 4; // 2^level+4

    /* Length of cell for this level */
    real celeng = L0 / POW2(level); // L0/2^level

    /* Inverse length of cell */
    real iL = ONE / celeng;

    /* Indices of the first and last real cells */
    int Asta = (POW8(level    ) - 1) / 7;
    int Aend = (POW8(level + 1) - 1) / 7 - 1;

    /* Starting index of Manother for this level */
    int ManotherStart = m2l_aux_get_starting_index_of_Manother_for_ij_blocking(r, level_start, level);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    /* Loop over real cells in this level */
    for (int A = Asta; A <= Aend; A ++) {
      /* Compute the coordinates of A */
      int itmp = POW2(level - 1);
      int ix = (int)FLOOR((center[A].x - center[0].x) * iL) + itmp; // 0<=ix<2^level
      int iy = (int)FLOOR((center[A].y - center[0].y) * iL) + itmp; // 0<=iy<2^level
      int iz = (int)FLOOR((center[A].z - center[0].z) * iL) + itmp; // 0<=iz<2^level
      /* Translate the origin by two ghost cells */
      ix += 2; // 2<=ix<2^level+2
      iy += 2; // 2<=iy<2^level+2
      iz += 2; // 2<=iz<2^level+2
      /* Copy original to another */
      int ncpec = ncpe / 2; /* 2^{level-1}+2 */
      int sib = 4 * (ix % 2) + 2 * (iy % 2) + (iz % 2); // sibling index
      int jx = ix / 2;
      int jy = iy / 2;
      int jz = iz / 2;
      for (int col = 0; col < r; col ++) { // column
	Manother[ManotherStart + (((col * 8 + sib) * ncpec + jz) * ncpec + jy) * ncpec + jx] = M[r * A + col];
      }	  
    }
  }
}


void m2l_aux_convert_M_to_Manother_for_ij_blocking_col1(int r, int level_start, int level_end, real3 *center, real L0, real *M, real *Manother)
{
  m2l_aux_convert_M_to_Manother_for_ij_blocking(r, level_start, level_end, center, L0, M, Manother);
}


void m2l_aux_convert_M_to_Manother_for_ij_blocking_col2(int r, int level_start, int level_end, real3 *center, real L0, real *M, real2 *Manother)
{
  /* (input) r: Dimension of M-vectors. level_start: First level where
     the ij-blocking scheme is applied. level_end: Last level where
     the ij-blocking scheme is applied. center: 3D-vector-type array
     of size ncell (ncell stands for the number of all the cells in
     all the levels). center[i].{x,y,z} denote the central coordinates
     of the i-th cell. L0: Length of the root cell (0th cell).  M:
     real-type array of size r*ncell. This contains M-vectors for all
     the cells in the standard format. (output) Manother:
     2D-vector-type array of size
     (r/2)*\sum_{l=level_start}^{level_end}(2^l+4)^3. This must be
     initilized beforehand. This will contain M-vectors for all the
     real and ghost cells in the special format for ij-blocking
     scheme. */

  /* Assertion */
  ASSERT(r % 2 == 0);

  /* Loop over levels to which ij-blocking scheme is applied */
  for (int level = level_start; level <= level_end; level ++) {

    /* Number of real and ghost cells per edge */
    int ncpe = POW2(level) + 4; // 2^level+4

    /* Length of cell for this level */
    real celeng = L0 / POW2(level); // L0/2^level

    /* Inverse length of cell */
    real iL = ONE / celeng;

    /* Indices of the first and last real cells */
    int Asta = (POW8(level    ) - 1) / 7;
    int Aend = (POW8(level + 1) - 1) / 7 - 1;

    /* Starting index of Manother for this level */
    int ManotherStart = m2l_aux_get_starting_index_of_Manother_for_ij_blocking(r, level_start, level);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    /* Loop over real cells in this level */
    for (int A = Asta; A <= Aend; A ++) {
      /* Compute the coordinates of A */
      int itmp = POW2(level - 1);
      int ix = (int)FLOOR((center[A].x - center[0].x) * iL) + itmp; // 0<=ix<2^level
      int iy = (int)FLOOR((center[A].y - center[0].y) * iL) + itmp; // 0<=iy<2^level
      int iz = (int)FLOOR((center[A].z - center[0].z) * iL) + itmp; // 0<=iz<2^level
      /* Translate the origin by two ghost cells */
      ix += 2; // 2<=ix<2^level+2
      iy += 2; // 2<=iy<2^level+2
      iz += 2; // 2<=iz<2^level+2
      /* Copy original to another */
      int ncpec = ncpe / 2; /* 2^{level-1}+2 */
      int sib = 4 * (ix % 2) + 2 * (iy % 2) + (iz % 2); // sibling index
      int jx = ix / 2;
      int jy = iy / 2;
      int jz = iz / 2;
      for (int col = 0; col < r; col += 2) { // column; unrolling 2x
	int loc = (ManotherStart / 2) + ((((col / 2) * 8 + sib) * ncpec + jz) * ncpec + jy) * ncpec + jx;
	Manother[loc].x = M[r * A + (col + 0)];
	Manother[loc].y = M[r * A + (col + 1)];
      }	  
    }
  }
}


void m2l_aux_convert_M_to_Manother_for_ij_blocking_col4(int r, int level_start, int level_end, real3 *center, real L0, real *M, real4 *Manother)
{
  /* (input) r: Dimension of M-vectors. level_start: First level where
     the ij-blocking scheme is applied. level_end: Last level where
     the ij-blocking scheme is applied. center: 3D-vector-type array
     of size ncell (ncell stands for the number of all the cells in
     all the levels). center[i].{x,y,z} denote the central coordinates
     of the i-th cell. L0: Length of the root cell (0th cell).  M:
     real-type array of size r*ncell. This contains M-vectors for all
     the cells in the standard format. (output) Manother:
     4D-vector-type array of size
     (r/4)*\sum_{l=level_start}^{level_end}(2^l+4)^3. This must be
     initilized beforehand. This will contain M-vectors for all the
     real and ghost cells in the special format for ij-blocking
     scheme. */

  /* Assertion */
  ASSERT(r % 4 == 0);

  /* Loop over levels to which ij-blocking scheme is applied */
  for (int level = level_start; level <= level_end; level ++) {

    /* Number of real and ghost cells per edge */
    int ncpe = POW2(level) + 4; // 2^level+4

    /* Length of cell for this level */
    real celeng = L0 / POW2(level); // L0/2^level

    /* Inverse length of cell */
    real iL = ONE / celeng;

    /* Indices of the first and last real cells */
    int Asta = (POW8(level    ) - 1) / 7;
    int Aend = (POW8(level + 1) - 1) / 7 - 1;

    /* Starting index of Manother for this level */
    int ManotherStart = m2l_aux_get_starting_index_of_Manother_for_ij_blocking(r, level_start, level);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    /* Loop over real cells in this level */
    for (int A = Asta; A <= Aend; A ++) {
      /* Compute the coordinates of A */
      int itmp = POW2(level - 1);
      int ix = (int)FLOOR((center[A].x - center[0].x) * iL) + itmp; // 0<=ix<2^level
      int iy = (int)FLOOR((center[A].y - center[0].y) * iL) + itmp; // 0<=iy<2^level
      int iz = (int)FLOOR((center[A].z - center[0].z) * iL) + itmp; // 0<=iz<2^level
      /* Translate the origin by two ghost cells */
      ix += 2; // 2<=ix<2^level+2
      iy += 2; // 2<=iy<2^level+2
      iz += 2; // 2<=iz<2^level+2
      /* Copy original to another */
      int ncpec = ncpe / 2; /* 2^{level-1}+2 */
      int sib = 4 * (ix % 2) + 2 * (iy % 2) + (iz % 2); // sibling index
      int jx = ix / 2;
      int jy = iy / 2;
      int jz = iz / 2;
      for (int col = 0; col < r; col += 4) { // column; unrolling 4x
	int loc = (ManotherStart / 4) + ((((col / 4) * 8 + sib) * ncpec + jz) * ncpec + jy) * ncpec + jx;
	Manother[loc].x = M[r * A + (col + 0)];
	Manother[loc].y = M[r * A + (col + 1)];
	Manother[loc].z = M[r * A + (col + 2)];
	Manother[loc].w = M[r * A + (col + 3)];
      }	  
    }
  }
}


void m2l_aux_convert_Lanother_to_L_for_ij_blocking(int r, real3 *center, real L0, int level, int B, real4 *Lanother, real *L)
{
  /* Assertion */
  ASSERT(r % 4 == 0);

  /* Length of cell for this level */
  real celeng = L0 / POW2(level); // L0/2^level
  
  /* Inverse length of cell */
  real iL = ONE / celeng;
  
  /* Indices of the first and last real cells */
  int Asta = (POW8(level    ) - 1) / 7;
  int Aend = (POW8(level + 1) - 1) / 7 - 1;

  /* Dimensions of chunk is fixed to B */
  int Dx = B, Dy = B, Dz = B;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  /* Loop over real cells */
  for (int A = Asta; A <= Aend; A ++) {
    /* Compute the coordinates of A */
    int itmp = POW2(level - 1);
    int ix = (int)FLOOR((center[A].x - center[0].x) * iL) + itmp; // 0<=ix<2^level
    int iy = (int)FLOOR((center[A].y - center[0].y) * iL) + itmp; // 0<=iy<2^level
    int iz = (int)FLOOR((center[A].z - center[0].z) * iL) + itmp; // 0<=iz<2^level
    /* Compute the chunk's coordinates */
    int cx = ix / (2 * Dx); // 0<=cx<2^level/(2*Dx)
    int cy = iy / (2 * Dy); // 0<=cy<2^level/(2*Dy)
    int cz = iz / (2 * Dz); // 0<=cz<2^level/(2*Dz)
    /* Compute the index of the chunk */
    int ncx = POW2(level) / (2 * Dx);     // number of chunks per edge along x axis
    int ncy = POW2(level) / (2 * Dy);     // number of chunks per edge along y axis
    int cid = cx + ncx * (cy + ncy * cz); // 0<=cid<ncx*ncy*ncz
    /* Compute the sibling-index of A (0<=sib<8) */
    int sib = (A - Asta) % 8;
    /* Compute the cell index for the child (the cluster index) within the chunk */
    int qx = (ix - (2 * Dx) * cx) / 2; // 0<=qx<Dx
    int qy = (iy - (2 * Dy) * cy) / 2; // 0<=qy<Dy
    int qz = (iz - (2 * Dz) * cz) / 2; // 0<=qz<Dz
    int ld = qx + Dx * (qz + Dz * qy); // 0<=ld<Dx*Dy*Dz; NOT qx+Dx*(qy+Dy*qz).
    /* Copy */
    real *Ltmp = &(L[r * (A - Asta)]); // row=0
    for (int row = 0; row < r; row += 4) { // unrolling 4x
      real4 Lanothertmp = Lanother[((cid * (r / 4) + (row / 4)) * 8 + sib) * (Dx * Dy * Dz) + ld];
      Ltmp[row    ] = Lanothertmp.x;
      Ltmp[row + 1] = Lanothertmp.y;
      Ltmp[row + 2] = Lanothertmp.z;
      Ltmp[row + 3] = Lanothertmp.w;
    }
  }
}


void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row1(int r, real3 *center, real L0, int level, int B, real *Lanother, real *L)
{
  //120214  /* Assertion */
  //120214  ASSERT(r % 2 == 0);

  /* Length of cell for this level */
  real celeng = L0 / POW2(level); // L0/2^level
  
  /* Inverse length of cell */
  real iL = ONE / celeng;
  
  /* Indices of the first and last real cells */
  int Asta = (POW8(level    ) - 1) / 7;
  int Aend = (POW8(level + 1) - 1) / 7 - 1;

  /* Dimensions of chunk is fixed to B */
  int Dx = B, Dy = B, Dz = B;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  /* Loop over real cells */
  for (int A = Asta; A <= Aend; A ++) {
    /* Compute the coordinates of A */
    int itmp = POW2(level - 1);
    int ix = (int)FLOOR((center[A].x - center[0].x) * iL) + itmp; // 0<=ix<2^level
    int iy = (int)FLOOR((center[A].y - center[0].y) * iL) + itmp; // 0<=iy<2^level
    int iz = (int)FLOOR((center[A].z - center[0].z) * iL) + itmp; // 0<=iz<2^level
    /* Compute the chunk's coordinates */
    int cx = ix / (2 * Dx); // 0<=cx<2^level/(2*Dx)
    int cy = iy / (2 * Dy); // 0<=cy<2^level/(2*Dy)
    int cz = iz / (2 * Dz); // 0<=cz<2^level/(2*Dz)
    /* Compute the index of the chunk */
    int ncx = POW2(level) / (2 * Dx);     // number of chunks per edge along x axis
    int ncy = POW2(level) / (2 * Dy);     // number of chunks per edge along y axis
    int cid = cx + ncx * (cy + ncy * cz); // 0<=cid<ncx*ncy*ncz
    /* Compute the sibling-index of A (0<=sib<8) */
    int sib = (A - Asta) % 8;
    /* Compute the cell index for the child (the cluster index) within the chunk */
    int qx = (ix - (2 * Dx) * cx) / 2; // 0<=qx<Dx
    int qy = (iy - (2 * Dy) * cy) / 2; // 0<=qy<Dy
    int qz = (iz - (2 * Dz) * cz) / 2; // 0<=qz<Dz
    int ld = qx + Dx * (qz + Dz * qy); // 0<=ld<Dx*Dy*Dz; NOT qx+Dx*(qy+Dy*qz).
    /* Copy */
    real *Ltmp = &(L[r * (A - Asta)]); // row=0
    for (int row = 0; row < r; row ++) { // no unrolling
      /////////////////////////////////////////////////////////////////////////////////////////////////////
      //      DBG("level=%d A=%d row=%d L=%15.7e\n", level, A, row, Lanother[((cid * r + row) * 8 + sib) * (Dx * Dy * Dz) + ld]);
      /////////////////////////////////////////////////////////////////////////////////////////////////////
      Ltmp[row] = Lanother[((cid * r + row) * 8 + sib) * (Dx * Dy * Dz) + ld];
    }
  }
}


void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row4(int r, real3 *center, real L0, int level, int B, real4 *Lanother, real *L)
{
  m2l_aux_convert_Lanother_to_L_for_ij_blocking(r, center, L0, level, B, Lanother, L);
}


void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row8(int r, real3 *center, real L0, int level, int B, real8 *Lanother, real *L)
{
  /* Assertion */
  ASSERT(r % 8 == 0);

  /* Length of cell for this level */
  real celeng = L0 / POW2(level); // L0/2^level
  
  /* Inverse length of cell */
  real iL = ONE / celeng;
  
  /* Indices of the first and last real cells */
  int Asta = (POW8(level    ) - 1) / 7;
  int Aend = (POW8(level + 1) - 1) / 7 - 1;

  /* Dimensions of chunk is fixed to B */
  int Dx = B, Dy = B, Dz = B;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  /* Loop over real cells */
  for (int A = Asta; A <= Aend; A ++) {
    /* Compute the coordinates of A */
    int itmp = POW2(level - 1);
    int ix = (int)FLOOR((center[A].x - center[0].x) * iL) + itmp; // 0<=ix<2^level
    int iy = (int)FLOOR((center[A].y - center[0].y) * iL) + itmp; // 0<=iy<2^level
    int iz = (int)FLOOR((center[A].z - center[0].z) * iL) + itmp; // 0<=iz<2^level
    /* Compute the chunk's coordinates */
    int cx = ix / (2 * Dx); // 0<=cx<2^level/(2*Dx)
    int cy = iy / (2 * Dy); // 0<=cy<2^level/(2*Dy)
    int cz = iz / (2 * Dz); // 0<=cz<2^level/(2*Dz)
    /* Compute the index of the chunk */
    int ncx = POW2(level) / (2 * Dx);     // number of chunks per edge along x axis
    int ncy = POW2(level) / (2 * Dy);     // number of chunks per edge along y axis
    int cid = cx + ncx * (cy + ncy * cz); // 0<=cid<ncx*ncy*ncz
    /* Compute the sibling-index of A (0<=sib<8) */
    int sib = (A - Asta) % 8;
    /* Compute the cell index for the child (the cluster index) within the chunk */
    int qx = (ix - (2 * Dx) * cx) / 2; // 0<=qx<Dx
    int qy = (iy - (2 * Dy) * cy) / 2; // 0<=qy<Dy
    int qz = (iz - (2 * Dz) * cz) / 2; // 0<=qz<Dz
    int ld = qx + Dx * (qz + Dz * qy); // 0<=ld<Dx*Dy*Dz; NOT qx+Dx*(qy+Dy*qz).
    /* Copy */
    real *Ltmp = &(L[r * (A - Asta)]); // row=0
    for (int row = 0; row < r; row += 8) { // unrolling 8x
      real8 Lanothertmp = Lanother[((cid * (r / 8) + (row / 8)) * 8 + sib) * (Dx * Dy * Dz) + ld];
      Ltmp[row    ] = Lanothertmp.a;
      Ltmp[row + 1] = Lanothertmp.b;
      Ltmp[row + 2] = Lanothertmp.c;
      Ltmp[row + 3] = Lanothertmp.d;
      Ltmp[row + 4] = Lanothertmp.e;
      Ltmp[row + 5] = Lanothertmp.f;
      Ltmp[row + 6] = Lanothertmp.g;
      Ltmp[row + 7] = Lanothertmp.h;
    }
  }
}


void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row16(int r, real3 *center, real L0, int level, int B, real16 *Lanother, real *L)
{
  /* Assertion */
  ASSERT(r % 16 == 0);

  /* Length of cell for this level */
  real celeng = L0 / POW2(level); // L0/2^level
  
  /* Inverse length of cell */
  real iL = ONE / celeng;
  
  /* Indices of the first and last real cells */
  int Asta = (POW8(level    ) - 1) / 7;
  int Aend = (POW8(level + 1) - 1) / 7 - 1;

  /* Dimensions of chunk is fixed to B */
  int Dx = B, Dy = B, Dz = B;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  /* Loop over real cells */
  for (int A = Asta; A <= Aend; A ++) {
    /* Compute the coordinates of A */
    int itmp = POW2(level - 1);
    int ix = (int)FLOOR((center[A].x - center[0].x) * iL) + itmp; // 0<=ix<2^level
    int iy = (int)FLOOR((center[A].y - center[0].y) * iL) + itmp; // 0<=iy<2^level
    int iz = (int)FLOOR((center[A].z - center[0].z) * iL) + itmp; // 0<=iz<2^level
    /* Compute the chunk's coordinates */
    int cx = ix / (2 * Dx); // 0<=cx<2^level/(2*Dx)
    int cy = iy / (2 * Dy); // 0<=cy<2^level/(2*Dy)
    int cz = iz / (2 * Dz); // 0<=cz<2^level/(2*Dz)
    /* Compute the index of the chunk */
    int ncx = POW2(level) / (2 * Dx);     // number of chunks per edge along x axis
    int ncy = POW2(level) / (2 * Dy);     // number of chunks per edge along y axis
    int cid = cx + ncx * (cy + ncy * cz); // 0<=cid<ncx*ncy*ncz
    /* Compute the sibling-index of A (0<=sib<8) */
    int sib = (A - Asta) % 8;
    /* Compute the cell index for the child (the cluster index) within the chunk */
    int qx = (ix - (2 * Dx) * cx) / 2; // 0<=qx<Dx
    int qy = (iy - (2 * Dy) * cy) / 2; // 0<=qy<Dy
    int qz = (iz - (2 * Dz) * cz) / 2; // 0<=qz<Dz
    int ld = qx + Dx * (qz + Dz * qy); // 0<=ld<Dx*Dy*Dz; NOT qx+Dx*(qy+Dy*qz).
    /* Copy */
    real *Ltmp = &(L[r * (A - Asta)]); // row=0
    for (int row = 0; row < r; row += 16) { // unrolling 16x
      real16 Lanothertmp = Lanother[((cid * (r / 16) + (row / 16)) * 8 + sib) * (Dx * Dy * Dz) + ld];
      Ltmp[row     ] = Lanothertmp.a;
      Ltmp[row +  1] = Lanothertmp.b;
      Ltmp[row +  2] = Lanothertmp.c;
      Ltmp[row +  3] = Lanothertmp.d;
      Ltmp[row +  4] = Lanothertmp.e;
      Ltmp[row +  5] = Lanothertmp.f;
      Ltmp[row +  6] = Lanothertmp.g;
      Ltmp[row +  7] = Lanothertmp.h;
      Ltmp[row +  8] = Lanothertmp.i;
      Ltmp[row +  9] = Lanothertmp.j;
      Ltmp[row + 10] = Lanothertmp.k;
      Ltmp[row + 11] = Lanothertmp.l;
      Ltmp[row + 12] = Lanothertmp.m;
      Ltmp[row + 13] = Lanothertmp.n;
      Ltmp[row + 14] = Lanothertmp.o;
      Ltmp[row + 15] = Lanothertmp.p;
    }
  }
}


void m2l_aux_comp_cellsta_cellend(int *cellsta, int *cellend, int level)
{
  /* Compute the indices to specify the staring and final cells for
     each level */
  cellsta[0] = 0;
  cellend[0] = POW8(0) - 1;
  for (int k = 1; k <= level; k ++) {
    cellsta[k] = cellend[k - 1] + 1;
    cellend[k] = cellsta[k] + POW8(k) - 1;
  }
}


double m2l_aux_comp_kernel_performance_in_Gflops(int r, int level_start, int level_end, int *cellsta, int *cellend, int *iinter, double time)
{
  double flop_per_interaction = r * (2 * r - 1);
  double num_interactions = 0;
  for (int level = level_start; level <= level_end; level ++) {
    for (int A = cellsta[level]; A <= cellend[level]; A ++) {
      num_interactions += (double)iinter[A];
    }
  }
  double flop = flop_per_interaction * num_interactions;
  double perf = flop / time / 1000000000;
  return perf;
}


static int m2l_aux_get_source_cluster_index(real L, real3 *center, int P, int U)
{
  real h = L / TWO;
  real dx = center[U].x - center[P].x;
  real dy = center[U].y - center[P].y;
  real dz = center[U].z - center[P].z;
  int ix, iy, iz;

  if (dx < - h) {
    ix = 0;
  } else if (dx <  h) {
    ix = 1;
  } else {
    ix = 2;
  }
  if (dy < - h) {
    iy = 0;
  } else if (dy <  h) {
    iy = 1;
  } else {
    iy = 2;
  }
  if (dz < - h) {
    iz = 0;
  } else if (dz <  h) {
    iz = 1;
  } else {
    iz = 2;
  }
  return (ix + 3 * iy + 9 * iz); // [0,26]
}


#if defined(ENABLE_USE_PARENT_LEAVES_ARRAYS)
void m2l_aux_comp_sourceclusters(int minlev, int maxlev, real L0, real3 *center, int *parent, int *leaves, int *ineigh, int pitch_neighbors, int *neighbors, int pitch_sourceclusters, int *sourceclusters)
#else
void m2l_aux_comp_sourceclusters(int minlev, int maxlev, real L0, real3 *center, int *ineigh, int pitch_neighbors, int *neighbors, int pitch_sourceclusters, int *sourceclusters)
#endif
{
  /* Compute cellsta and cellend */
  int *cellsta = (int *)malloc((maxlev + 1) * sizeof(int)); // cellsta[0:maxlev]
  int *cellend = (int *)malloc((maxlev + 1) * sizeof(int)); // cellend[0:maxlev]
  m2l_aux_comp_cellsta_cellend(cellsta, cellend, maxlev);

  //111123#if(1)
  /* Loop over levels for clusters */
  for (int plevel = minlev - 1; plevel <= maxlev - 1; plevel ++) {

    /* Obtain the level of cells in clusters */
    const int level = plevel + 1;

    /* Compute the edge length of cells */
    const real length = L0 / POW2(level); // L0/(2^level)

    /* Loop over field clusters */
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int FC = cellsta[plevel]; FC <= cellend[plevel]; FC ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
      
      /* Obtain the index of FC when the 0th index is assigned to the
	 first FC in level minlev-1; 0<=F0<nallcluster */
      const int F0 = FC - cellsta[minlev - 1];

      /* Initialise FC's list */
      for (int i = 0; i < pitch_sourceclusters; i ++) {
	sourceclusters[pitch_sourceclusters * F0 + i] = NULL_CELL; // not FC, but F0
      }
      
      /* Loop over FC's neighbours */
      for (int i = 0; i < ineigh[FC]; i ++) {

	/* FC's neighbour SC */
	const int SC = neighbors[pitch_neighbors * FC + i];

	/* Exclude SC such as SC=FC */
	if (SC != FC) {

	  /* Obtain source-cluster index of SC */
	  const int d = m2l_aux_get_source_cluster_index(length * 2, center, FC, SC);
	
	  /* Store the 0th sibling of SC into the list of FC */
#if defined(ENABLE_USE_PARENT_LEAVES_ARRAYS)
	  sourceclusters[pitch_sourceclusters * F0 + d] = leaves[8 * SC + 0]; // not FC, but F0
#else
	  sourceclusters[pitch_sourceclusters * F0 + d] = GET_CHILD_INDEX(SC, 0); // not FC, but F0
#endif

	}
      }
    }
  }
  //111123#else
  //111123  /* Initialise the index of cluster */
  //111123  int FC = 0;
  //111123
  //111123  /* Loop over levels */
  //111123  for (int level = minlev; level <= maxlev; level ++) {
  //111123
  //111123    /* Compute the edge length of cells at this level */
  //111123    real length = L0 / POW2(level); // L0/(2^level)
  //111123
  //111123    /* Loop over field cells with sibling-index 0 */
  //111123    for (int A0 = cellsta[level]; A0 <= cellend[level]; A0 += 8, FC ++) {
  //111123      
  //111123      /* Initialise FC's list */
  //111123      for (int i = 0; i < pitch_sourceclusters; i ++) {
  //111123	sourceclusters[pitch_sourceclusters * FC + i] = NULL_CELL;
  //111123      }
  //111123      
  //111123      /* Obtain A0's parent P, which is the same as FC */
  //111123      int P = parent[A0];
  //111123
  //111123      /* Loop over P's neighbours */
  //111123      for (int i = 0; i < ineigh[P]; i ++) {
  //111123
  //111123	/* P's neighbour U */
  //111123	int U = neighbors[pitch_neighbors * P + i];
  //111123
  //111123	/* Exclude U such as U=P */
  //111123	if (U != P) {
  //111123
  //111123	  /* Obtain source-cluster index of U */
  //111123	  int d = m2l_aux_get_source_cluster_index(length * 2, center, P, U);
  //111123	
  //111123	  /* Store A0 into the list of FC */
  //111123	  sourceclusters[pitch_sourceclusters * FC + d] = leaves[8 * U + 0];
  //111123
  //111123	}
  //111123      }
  //111123    }
  //111123  }
  //111123#endif

  free(cellsta);
  free(cellend);
}


int m2l_aux_get_a_Kindex(int *Ktable, int vx, int vy, int vz)
{
  /* (input) Ktable: Output from m2l_aux_comp_Ktable.  vx, vy, vz:
     Integers in [-3,3].

     This function returns a K-index associated with a dimensionless
     M2L-vector (vx,vy,vz). If the M2L-vector points to a
     near-neighbor cell (i.e. (vx,vy,vz) in [-1,1]^3), the function
     returns NULL_KINDEX. */

  return Ktable[49 * (vx + 3) + 7 * (vy + 3) + (vz + 3)];
}


void m2l_aux_comp_Kindex(int maxlev, real L0, real3 *center, int *iinter, int pitch_interaction, int *interaction, int *Ktable, int *Kindex)
{
  /* Compute cellsta and cellend */
  int *cellsta = (int *)malloc((maxlev + 1) * sizeof(int)); // cellsta[0:maxlev]
  int *cellend = (int *)malloc((maxlev + 1) * sizeof(int)); // cellend[0:maxlev]
  m2l_aux_comp_cellsta_cellend(cellsta, cellend, maxlev);

  /* Loop over levels */
  for (int level = 0; level <= maxlev; level ++) {
    
    /* Compute the edge length of cells in this level */
    real L = L0 / POW2(level);

    /* Compute the inverse of L */
    real iL = ONE / L;

    /* Loop over field cells */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int F = cellsta[level]; F <= cellend[level]; F ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.

      /* Center of F */
      real3 fcenter = center[F];

      /* Loop over source cells */
      for (int i = 0; i < iinter[F]; i ++) {
	
	/* Load the i-th source cell S */
	int S = interaction[pitch_interaction * F + i];

	/* Center of S */
	real3 scenter = center[S];

	/* Normalize the vector from F to S by L */
	real3 diff;
	diff.x = (scenter.x - fcenter.x) * iL;
	diff.y = (scenter.y - fcenter.y) * iL;
	diff.z = (scenter.z - fcenter.z) * iL;

	/* Compute the nearest integer */
	int vx = (int)RINT(diff.x);
	int vy = (int)RINT(diff.y);
	int vz = (int)RINT(diff.z);

	/* Append the baseaddress to K */
	Kindex[pitch_interaction * F + i] = m2l_aux_get_a_Kindex(Ktable, vx, vy, vz);

      }
    }
  }

  /* Free */
  free(cellsta);
  free(cellend);

}
