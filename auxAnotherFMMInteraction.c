#include "auxAnotherFMMInteraction.h"

#if defined(TEST_M2L_AUXILIARY_C)
#include "m2l_auxiliary.h"
#endif

void aux_store_field_values(int dofn3, int pitch2, int Asta, int Aend, real *fieldval, real *F)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = Asta; A <= Aend; A ++) {
    for (int i = 0; i < dofn3; i ++) { // LOOP WAS VECTORIZED.
      fieldval[dofn3 * A + i] = F[pitch2 * (A - Asta) + i];
    }
  }
}


#if defined(TEST_M2L_AUXILIARY_C)

void aux_convert_E_cluster_blocking(int cutoff, int *Ktable, real *E, real *K)
{
  int P, Q;
  if (cutoff == 32) {
    P = 1;
    Q = 1;
  } else if (cutoff == 256) {
    P = 8;
    Q = 8;
  }
  m2l_aux_convert_K_to_Kanother_for_cluster_blocking(cutoff, P, Q, E, K);
}

#else

void aux_convert_E_cluster_blocking(int cutoff, int *Ktable, real *E, real *K)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int k = 0; k < 316; k ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    int count = 0;
    for (int q = 0; q < cutoff / 32; q ++) {
      for (int p = 0; p < cutoff / 32; p ++) {
	for (int col = 0; col < 32; col ++) {
	  int j = q * 32 + col;
	  for (int row = 0; row < 32; row ++) { // LOOP WAS VECTORIZED.
	    int i = p * 32 + row;	    
	    K[cutoff * cutoff * k + count] = E[cutoff * cutoff * k + i + cutoff * j];
	    count++;
	  }
	}
      }
    }
  }
}

#endif /* TEST_M2L_AUXILIARY_C */

#if defined(TEST_M2L_AUXILIARY_C)

void aux_convert_E_ij_blocking_real4(int cutoff, int *Ktable, real *E, void *Kanother)
{
  m2l_aux_convert_K_to_Kanother_for_ij_blocking(cutoff, Ktable, E, Kanother);
}

#else

void aux_convert_E_ij_blocking_real4(int cutoff, int *Ktable, real *E, void *Kanother)
{
  real4 *Kanother4 = (real4 *)Kanother; // cast
  int nnn = 0; // index for non near-neighbour matricies (0<=nnn<316)
  for (int kz = - 3; kz < 4; kz ++) { // order of loops is reverse to that for creating Ktable.
    for (int ky = - 3; ky < 4; ky ++) {
      for (int kx = - 3; kx < 4; kx ++) {
	int k = Ktable[49 * (kx + 3) + 7 * (ky + 3) + (kz + 3)]; // 0<=k<316
	if (k != - 1) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	  for (int j = 0; j < cutoff; j ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
	    for (int i = 0; i < cutoff; i += 4) { // unrolling 4
	      int itmpK = 316 * (i / 4 + (cutoff / 4) * j) + nnn;
	      int itmpE = i + cutoff * j + cutoff * cutoff * k;
	      Kanother4[itmpK].x = E[itmpE    ]; // row i
	      Kanother4[itmpK].y = E[itmpE + 1]; // row i+1
	      Kanother4[itmpK].z = E[itmpE + 2]; // row i+2
	      Kanother4[itmpK].w = E[itmpE + 3]; // row i+3
	    }
	  }
	  nnn ++;
	}
      }
    }
  }
}

#endif /* TEST_M2L_AUXILIARY_C */


void aux_convert_E_ij_blocking_real2(int cutoff, int *Ktable, real *E, void *Kanother)
{
  real2 *Kanother2 = (real2 *)Kanother; // cast
  int nnn = 0; // index for non near-neighbour matricies (0<=nnn<316)
  for (int kz = - 3; kz < 4; kz ++) { // order of loops is reverse to that for creating Ktable.
    for (int ky = - 3; ky < 4; ky ++) {
      for (int kx = - 3; kx < 4; kx ++) {
	int k = Ktable[49 * (kx + 3) + 7 * (ky + 3) + (kz + 3)]; // 0<=k<316
	if (k != - 1) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	  for (int j = 0; j < cutoff; j ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
	    for (int i = 0; i < cutoff; i += 2) { // unrolling 2
	      int itmpK = 316 * (i / 2 + (cutoff / 2) * j) + nnn;
	      int itmpE = i + cutoff * j + cutoff * cutoff * k;
	      Kanother2[itmpK].x = E[itmpE    ]; // row i
	      Kanother2[itmpK].y = E[itmpE + 1]; // row i+1
	    }
	  }
	  nnn ++;
	}
      }
    }
  }
}


void aux_convert_E_ij_blocking(int cutoff, int *Ktable, real *E, real *Kanother)
{
  int nnn = 0; // index for non near-neighbour matricies (0<=nnn<316)
  for (int kz = - 3; kz < 4; kz ++) { // order of loops is reverse to that for creating Ktable.
    for (int ky = - 3; ky < 4; ky ++) {
      for (int kx = - 3; kx < 4; kx ++) {
	int k = Ktable[49 * (kx + 3) + 7 * (ky + 3) + (kz + 3)]; // 0<=k<316
	if (k != - 1) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
	  for (int j = 0; j < cutoff; j ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
#if(1)
	    real *pKanother = &(Kanother[316 * (0 + cutoff * j) + nnn]); // point to i=0
	    real *pE = &(E[0 + cutoff * j + cutoff * cutoff * k]); // point to i=0
	    for (int i = 0; i < cutoff; i ++) {
	      pKanother[316 * i] = pE[i];
	    }
#else
	    for (int i = 0; i < cutoff; i ++) {
	      Kanother[316 * (i + cutoff * j) + nnn] = E[i + cutoff * j + cutoff * cutoff * k];
	    }
#endif
	  }
	  nnn ++;
	}
      }
    }
  }
}


#if defined(TEST_M2L_AUXILIARY_C)

void aux_convert_PS_ij_blocking(int cutoff, int *levsta, int *levend, real3 *center, real *celeng,
				int switchlevel_high, int maxlev, int *PSanotherstart,
				real *PS, real *PSanother)
{
  m2l_aux_convert_M_to_Manother_for_ij_blocking(cutoff, switchlevel_high + 1, maxlev, center, celeng[0], PS, PSanother);
}

#else

void aux_convert_PS_ij_blocking(int cutoff, int *levsta, int *levend, real3 *center, real *celeng,
				int switchlevel_high, int maxlev, int *PSanotherstart,
				real *PS, real *PSanother)
{
  for (int level = switchlevel_high + 1; level <= maxlev; level ++) {
    /* Number of cells (including four ghost cells) per edge */
    int ncpe = POW2(level) + 4; // 2^level+4
    /* Inverse length of cell */
    real iL = ONE / celeng[level];
    /* Loop over real cells in this level */
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
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
      int ctype = 4 * (ix % 2) + 2 * (iy % 2) + (iz % 2);
      int jx = ix / 2;
      int jy = iy / 2;
      int jz = iz / 2;
#if(1)
      real *pPSanother = &(PSanother[PSanotherstart[level] + (((0 * 8 + ctype) * ncpec + jz) * ncpec + jy) * ncpec + jx]); // point to col=0
      real *pPS = &(PS[cutoff * A + 0]); // point to col=0
      int ncpe3 = 8 * ncpec * ncpec * ncpec; // ncpe*ncpe*ncpe
      for (int col = 0; col < cutoff; col ++) { // column
	pPSanother[col * ncpe3] = pPS[col];
      }	  
#else
      for (int col = 0; col < cutoff; col ++) { // column
	PSanother[PSanotherstart[level] + (((col * 8 + ctype) * ncpec + jz) * ncpec + jy) * ncpec + jx] = PS[cutoff * A + col];
      }	  
#endif
    }
  }
}

#endif /* TEST_M2L_AUXILIARY_C */


#if defined(TEST_M2L_AUXILIARY_C)

void aux_convert_PF_ij_blocking_real4(int cutoff, int *levsta, int *levend, real iL, real3 *center,
					int level, int Dx, int Dy, int Dz, real *PFtmp, void *PF)
{
  ASSERT(Dx == Dy && Dy == Dz);

  real L0 = (ONE / iL) * POW2(level); // (1/(2^l/L0))*2^l

  m2l_aux_convert_Lanother_to_L_for_ij_blocking(cutoff, center, L0, level, Dx, PF, PFtmp);
}

#else

void aux_convert_PF_ij_blocking_real4(int cutoff, int *levsta, int *levend, real iL, real3 *center,
					int level, int Dx, int Dy, int Dz, real *PFtmp, void *PF)
{
  real4 *PF4 = (real4 *)PF; // cast
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    /* Compute the coordinates of A */
    int itmp = POW2(level - 1);
    int ix = (int)FLOOR((center[A].x - center[0].x) * iL) + itmp; // 0<=ix<2^level
    int iy = (int)FLOOR((center[A].y - center[0].y) * iL) + itmp; // 0<=iy<2^level
    int iz = (int)FLOOR((center[A].z - center[0].z) * iL) + itmp; // 0<=iz<2^level
    /* Compute the chunk's coordinates */
    int cx = ix / (2 * Dx); // 0<=cx<2^level/(2*Dx)
    int cy = iy / (2 * Dy); // 0<=cy<2^level/(2*Dy)
    int cz = iz / (2 * Dz); // 0<=cz<2^level/(2*Dz)
    /* Compute the chunk's ID */
    int ncx = POW2(level) / (2 * Dx); // number of chunks per edge along x axis
    int ncy = POW2(level) / (2 * Dy); // number of chunks per edge along y axis
    int cid = cx + ncx * (cy + ncy * cz); // 0<=cid<ncx*ncy*ncz
    /* Compute the child-type of A (0<=ctype<8)*/
    int ctype = (A - levsta[level]) % 8;
    /* Compute the cell index for the child (the cluster index) within the chunk */
    int qx = (ix - (2 * Dx) * cx) / 2; // 0<=qx<Dx
    int qy = (iy - (2 * Dy) * cy) / 2; // 0<=qy<Dy
    int qz = (iz - (2 * Dz) * cz) / 2; // 0<=qz<Dz
    int ld = qx + Dx * (qz + Dz * qy); // 0<=ld<Dx*Dy*Dz; NOT qx+Dx*(qy+Dy*qz).
    /* Copy */
    real *PFtmpA = &(PFtmp[cutoff * (A - levsta[level])]); // row=0
    for (int row = 0; row < cutoff; row += 4) { // unrlloing 4
      real4 tmp4 = PF4[((cid * (cutoff / 4) + (row / 4)) * 8 + ctype) * (Dx * Dy * Dz) + ld];
      PFtmpA[row    ] = tmp4.x;
      PFtmpA[row + 1] = tmp4.y;
      PFtmpA[row + 2] = tmp4.z;
      PFtmpA[row + 3] = tmp4.w;
    }
  }
}

#endif /* TEST_M2L_AUXILIARY_C */


void aux_convert_PF_ij_blocking_real2(int cutoff, int *levsta, int *levend, real iL, real3 *center,
					int level, int Dx, int Dy, int Dz, real *PFtmp, void *PF)
{
  real2 *PF2 = (real2 *)PF; // cast
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    /* Compute the coordinates of A */
    int itmp = POW2(level - 1);
    int ix = (int)FLOOR((center[A].x - center[0].x) * iL) + itmp; // 0<=ix<2^level
    int iy = (int)FLOOR((center[A].y - center[0].y) * iL) + itmp; // 0<=iy<2^level
    int iz = (int)FLOOR((center[A].z - center[0].z) * iL) + itmp; // 0<=iz<2^level
    /* Compute the chunk's coordinates */
    int cx = ix / (2 * Dx); // 0<=cx<2^level/(2*Dx)
    int cy = iy / (2 * Dy); // 0<=cy<2^level/(2*Dy)
    int cz = iz / (2 * Dz); // 0<=cz<2^level/(2*Dz)
    /* Compute the chunk's ID */
    int ncx = POW2(level) / (2 * Dx); // number of chunks per edge along x axis
    int ncy = POW2(level) / (2 * Dy); // number of chunks per edge along y axis
    int cid = cx + ncx * (cy + ncy * cz); // 0<=cid<ncx*ncy*ncz
    /* Compute the child-type of A (0<=ctype<8)*/
    int ctype = (A - levsta[level]) % 8;
    /* Compute the cell index for the child (the cluster index) within the chunk */
    int qx = (ix - (2 * Dx) * cx) / 2; // 0<=qx<Dx
    int qy = (iy - (2 * Dy) * cy) / 2; // 0<=qy<Dy
    int qz = (iz - (2 * Dz) * cz) / 2; // 0<=qz<Dz
    int ld = qx + Dx * (qz + Dz * qy); // 0<=ld<Dx*Dy*Dz; NOT qx+Dx*(qy+Dy*qz).
    /* Copy */
    real *PFtmpA = &(PFtmp[cutoff * (A - levsta[level])]); // row=0
    for (int row = 0; row < cutoff; row += 2) { // unrlloing 2
      real2 tmp2 = PF2[((cid * (cutoff / 2) + (row / 2)) * 8 + ctype) * (Dx * Dy * Dz) + ld];
      PFtmpA[row    ] = tmp2.x;
      PFtmpA[row + 1] = tmp2.y;
    }
  }
}


void aux_convert_PF_ij_blocking(int cutoff, int *levsta, int *levend, real iL, real3 *center,
				int level, int Dx, int Dy, int Dz, real *PFtmp, real *PF)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    /* Compute the coordinates of A */
    int itmp = POW2(level - 1);
    int ix = (int)FLOOR((center[A].x - center[0].x) * iL) + itmp; // 0<=ix<2^level
    int iy = (int)FLOOR((center[A].y - center[0].y) * iL) + itmp; // 0<=iy<2^level
    int iz = (int)FLOOR((center[A].z - center[0].z) * iL) + itmp; // 0<=iz<2^level
    /* Compute the chunk's coordinates */
    int cx = ix / (2 * Dx); // 0<=cx<2^level/(2*Dx)
    int cy = iy / (2 * Dy); // 0<=cy<2^level/(2*Dy)
    int cz = iz / (2 * Dz); // 0<=cz<2^level/(2*Dz)
    /* Compute the chunk's ID */
    int ncx = POW2(level) / (2 * Dx); // number of chunks per edge along x axis
    int ncy = POW2(level) / (2 * Dy); // number of chunks per edge along y axis
    int cid = cx + ncx * (cy + ncy * cz); // 0<=cid<ncx*ncy*ncz
    /* Compute the child-type of A (0<=ctype<8)*/
    int ctype = (A - levsta[level]) % 8;
    /* Compute the cell index for the child (the cluster index) within the chunk */
    int qx = (ix - (2 * Dx) * cx) / 2; // 0<=qx<Dx
    int qy = (iy - (2 * Dy) * cy) / 2; // 0<=qy<Dy
    int qz = (iz - (2 * Dz) * cz) / 2; // 0<=qz<Dz
    int ld = qx + Dx * (qz + Dz * qy); // 0<=ld<Dx*Dy*Dz; NOT qx+Dx*(qy+Dy*qz).
    /* Copy */
#if(1)
    real *pPFtmp = &(PFtmp[cutoff * (A - levsta[level]) + 0]); // pointer to row=0
    real *pPF = &(PF[((cid * cutoff + 0) * 8 + ctype) * (Dx * Dy * Dz) + ld]); // pointer to row=0
    int Dxyz8 = 8 * Dx * Dy * Dz;
    for (int row = 0; row < cutoff; row ++) {
      pPFtmp[row] = pPF[row * Dxyz8];
    }
#else
    for (int row = 0; row < cutoff; row ++) {
      PFtmp[cutoff * (A - levsta[level]) + row] = PF[((cid * cutoff + row) * 8 + ctype) * (Dx * Dy * Dz) + ld];
    }
#endif
  }  
}

/******************************************
  Functions to run postM2L on CPU
******************************************/

void postm2l_convert_U_from_column_major_to_row_major(real *Ucol, real *Urow, int dofn3, int cutoff)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < dofn3; i ++) { // row; OpenMP DEFINED LOOP WAS PARALLELIZED.
    for (int j = 0; j < cutoff; j ++) { // column
      Urow[cutoff * i + j] = Ucol[i + dofn3 * j];
    }
  }
}

void postm2l_compute_adjusting_vector(real *Kweights, real *adjust, int dof, int n3)
{
  if (dof == 1) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < n3; i ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
      adjust[i] = Kweights[i];
    }
  } else {
    int l = 0;
    for (int i = 0; i < n3; i ++) { // impossilbe to be parallelised because of the variable l
      real tmp = Kweights[i];
      for (int j = 0; j < dof; j ++) { // LOOP WAS VECTORIZED.
	adjust[l] = tmp;
	l ++;
      }
    }
  }
}


void postm2l(int Asta, int Aend, int dofn3, int cutoff,
	     real *Urow, real *PF, real scale, real *adjust, real *F)
{
  /* Loop over cells in the underlying cell group */
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = Asta; A <= Aend; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    
    /* Compute A's field values F=diag(adjust)*U*PF */
    for (int i = 0; i < dofn3; i ++) { // row; OpenMP DEFINED LOOP WAS PARALLELIZED.
      real sum = ZERO;
      for (int j = 0; j < cutoff; j ++) { // column; LOOP WAS VECTORIZED.
	sum += Urow[cutoff * i + j] * PF[cutoff * (A - Asta) + j];
      }
      F[dofn3 * A + i] = scale * adjust[i] * sum;
    }
  }
}
