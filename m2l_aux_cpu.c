#include "m2l_aux_cpu.h"

void m2l_aux_convert_M_to_Manother_for_ij_blocking_col1_CPU(int r, int level_start, int level_end, real3 *center, real L0, real *M, real *Manother)
{
  /* Convert to Manother[level][j][2^l+4][2^l+4][2^l+4] regardless of
     sibling index */

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
      //120218      int ncpec = ncpe / 2; /* 2^{level-1}+2 */
      //120218      int sib = 4 * (ix % 2) + 2 * (iy % 2) + (iz % 2); // sibling index
      //120218      int jx = ix / 2;
      //120218      int jy = iy / 2;
      //120218      int jz = iz / 2;
      for (int col = 0; col < r; col ++) { // column
	//120218	Manother[ManotherStart + (((col * 8 + sib) * ncpec + jz) * ncpec + jy) * ncpec + jx] = M[r * A + col];
	Manother[ManotherStart + ((col * ncpe + iz) * ncpe + iy) * ncpe + ix] = M[r * A + col];
      }	  
    }
  }
}


void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row1_CPU(int r, real3 *center, real L0, int level, int B, real *Lanother, real *L)
{
  /* Length of cell for this level */
  real celeng = L0 / POW2(level); // L0/2^level
  
  /* Inverse length of cell */
  real iL = ONE / celeng;
  
  /* Indices of the first and last real cells */
  int Asta = (POW8(level    ) - 1) / 7;
  int Aend = (POW8(level + 1) - 1) / 7 - 1;

  //  /* Dimensions of chunk is fixed to B */
  //  int Dx = B, Dy = B, Dz = B;

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
    //    int cx = ix / (2 * Dx); // 0<=cx<2^level/(2*Dx)
    //    int cy = iy / (2 * Dy); // 0<=cy<2^level/(2*Dy)
    //    int cz = iz / (2 * Dz); // 0<=cz<2^level/(2*Dz)
    int cx = ix / (2 * B); // 0<=cx<2^level/(2*B)
    int cy = iy / (2 * B); // 0<=cy<2^level/(2*B)
    int cz = iz / (2 * B); // 0<=cz<2^level/(2*B)
    /* Compute the index of the chunk */
    //    int ncx = POW2(level) / (2 * Dx);     // number of chunks per edge along x axis
    //    int ncy = POW2(level) / (2 * Dy);     // number of chunks per edge along y axis
    int ncx = POW2(level) / (2 * B);     // number of chunks per edge along x axis
    int ncy = POW2(level) / (2 * B);     // number of chunks per edge along y axis
    int cid = cx + ncx * (cy + ncy * cz); // 0<=cid<ncx*ncy*ncz
    /* Compute the sibling-index of A (0<=sib<8) */
    int sib = (A - Asta) % 8;
    /* Compute the cell index for the child (the cluster index) within the chunk */
    //    int qx = (ix - (2 * Dx) * cx) / 2; // 0<=qx<Dx
    //    int qy = (iy - (2 * Dy) * cy) / 2; // 0<=qy<Dy
    //    int qz = (iz - (2 * Dz) * cz) / 2; // 0<=qz<Dz
    int qx = (ix - (2 * B) * cx) / 2; // 0<=qx<B
    int qy = (iy - (2 * B) * cy) / 2; // 0<=qy<B
    int qz = (iz - (2 * B) * cz) / 2; // 0<=qz<B
    //    int ld = qx + Dx * (qz + Dz * qy); // 0<=ld<Dx*Dy*Dz; NOT qx+Dx*(qy+Dy*qz).
    /* Copy */
    real *Ltmp = &(L[r * (A - Asta)]); // row=0
    //    for (int row = 0; row < r; row ++) { // no unrolling
    //      Ltmp[row] = Lanother[((cid * r + row) * 8 + sib) * (Dx * Dy * Dz) + ld];
    for (int i = 0; i < r; i ++) { // no unrolling
      Ltmp[i] = Lanother[((((cid * r + i) * B + qz) * B + qy) * B + qx) * 8 + sib];
    }
  }
}

void m2l_aux_convert_Lanother_to_L_for_ij_blocking_row1_CPU2(int r, real3 *center, real L0, int level, int B, real *Lanother, real *L)
{
  /* Length of cell for this level */
  real celeng = L0 / POW2(level); // L0/2^level
  
  /* Inverse length of cell */
  real iL = ONE / celeng;
  
  /* Indices of the first and last real cells */
  int Asta = (POW8(level    ) - 1) / 7;
  int Aend = (POW8(level + 1) - 1) / 7 - 1;

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
    int cx = ix / (2 * B); // 0<=cx<2^level/(2*B)
    int cy = iy / (2 * B); // 0<=cy<2^level/(2*B)
    int cz = iz / (2 * B); // 0<=cz<2^level/(2*B)
    /* Compute the index of the chunk */
    int ncx = POW2(level) / (2 * B);     // number of chunks per edge along x axis
    int ncy = POW2(level) / (2 * B);     // number of chunks per edge along y axis
    int cid = cx + ncx * (cy + ncy * cz); // 0<=cid<ncx*ncy*ncz
    /* Compute the sibling-index of A (0<=sib<8) */
    int sib = (A - Asta) % 8;
    /* Compute the cell index for the child (the cluster index) within the chunk */
    int qx = (ix - (2 * B) * cx) / 2; // 0<=qx<B
    int qy = (iy - (2 * B) * cy) / 2; // 0<=qy<B
    int qz = (iz - (2 * B) * cz) / 2; // 0<=qz<B
    /* Copy */
    real *Ltmp = &(L[r * (A - Asta)]); // row=0
    for (int i = 0; i < r; i ++) { // no unrolling
      //      Ltmp[i] = Lanother[((((cid * r + i) * B + qz) * B + qy) * B + qx) * 8 + sib];
      Ltmp[i] = Lanother[((((cid * r + i) * 8 + sib) * B + qz) * B + qy) * B + qx];
    }
  }
}
