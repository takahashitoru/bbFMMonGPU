#include <stdio.h>
#include <stdlib.h>

/* based on aux_CPU9A.c */
/* same as aux_scuda45I.c */
/* based on aux_scuda38BH.c */

//#define B 4
#define B 2

int main()
{
  int Ktable[343];
  // Initialize lookup table
  for (int i=0;i<343;i++)
    Ktable[i] = -1;
  
  // Create lookup table
  int ncell = 0;
  int ninteract = 0;
  for (int kz = - 3; kz < 4; kz ++) {
    for (int ky = - 3; ky < 4; ky ++) {
      for (int kx = - 3; kx < 4; kx ++) {
	if (abs(kx) > 1 || abs(ky) > 1 || abs(kz) > 1) {
	  Ktable[ncell] = ninteract;
	  ninteract ++;
	}
	ncell ++;
      }
    }
  }	

  int Kijoff_diff[8][189], Mjoff_diff[8][189];

  /* Loop over observation cell's sibling-index */
  for (int tz = 0; tz < 8; tz ++) {

    //120217    printf("#define B%d_COMPXYZ%d() ", B, tz);

    /* Compute the obs's sibling-index coordinates: 0<=lx,ly,lz<2 */
    int lx = (tz >> 2);       // tz/4
    int ly = ((tz & 3) >> 1); // (tz%4)/2
    int lz = (tz & 1);        // tz%2

    /* Initialise offsets */
    int Mjoff = 0;
    int Kijoff = 0;

    int nnn = 0;

    /* Loop over the shifted M2L-vectors: fdx=fx+lx+2 and fx=-2-lx to
       3-lx, thus fdx=0 to 5, where these six does not stand for B+2
       but the dimension of the near-neighbour  */
    for (int fdz = 0; fdz < 6; fdz ++) {
      for (int fdy = 0; fdy < 6; fdy ++) {
	for (int fdx = 0; fdx < 6; fdx ++) {
	  
	  /* Obtain the M2L-vector */
	  int kx = fdx - 2 - lx; // -2-lx<=kx<=3-lx
	  int ky = fdy - 2 - ly;
	  int kz = fdz - 2 - lz;

	  /* Screen near-neighbour source cells */
	  if (abs(kx) > 1 || abs(ky) > 1 || abs(kz) > 1) {

	    /* Obtain the source's sibling-index */
	    int sib = 4 * (fdx % 2) + 2 * (fdy % 2) + (fdz % 2);

	    /* Obtain the source's coordinates in 6x6x6 region (except
	       for the offset due to obs's sibling-index) */
	    int ix = fdx / 2; // 0<=ix<3
	    int iy = fdy / 2; // 0<=iy<3
	    int iz = fdz / 2; // 0<=iz<3

	    int Mjoff_old = Mjoff;
	    int Kijoff_old = Kijoff;
	    Mjoff = ((sib * (B + 2) + iz) * (B + 2) + iy) * (B + 2) + ix; // Mj[8][B+2][B+2][B+2]
	    Kijoff = Ktable[(kx + 3) + 7 * (ky + 3) + 49 * (kz + 3)];

	    Mjoff_diff[tz][nnn] = Mjoff - Mjoff_old;
	    Kijoff_diff[tz][nnn] = Kijoff - Kijoff_old;
	    nnn ++;

	    //120217	    printf("COMP(%d, %d)", Kijoff - Kijoff_old, Mjoff - Mjoff_old);
	    //120217	    if (!(fdz == 5 && fdy == 5 && fdx == 5)) {
	    //120217	      printf("; ");
	    //120217	    }

	  }
	}
      }
    }

    if (nnn != 189) {
      fprintf(stderr, "Something wrong\n");
      exit(EXIT_FAILURE);
    }

    //120217    printf("\n");
  }

  printf("#define CMP%d(i, M0, M1, M2, M3, M4, M5, M6, M7, K0, K1, K2, K3, K4, K5, K6, K7) \\\n", B);
  printf("  Mjptr[0] += M0; Mjptr[1] += M1; Mjptr[2] += M2; Mjptr[3] += M3; Mjptr[4] += M4; Mjptr[5] += M5; Mjptr[6] += M6; Mjptr[7] += M7; \\\n");
  printf("  Kijptr[0] += K0; Kijptr[1] += K1; Kijptr[2] += K2; Kijptr[3] += K3; Kijptr[4] += K4; Kijptr[5] += K5; Kijptr[6] += K6; Kijptr[7] += K7; \\\n");
  printf("  for (int s = 0; s < 8; s ++) { \\\n");
  printf("    Lij[s] += (*Kijptr[s]) * (*Mjptr[s]); \\\n");
  printf("  }\n");

  //  printf("#define CMP_B%d {", B);
  for (int i = 0; i < 189; i ++) {
    printf("CMP%d(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d); ",
	   B, i, Mjoff_diff[0][i], Mjoff_diff[1][i], Mjoff_diff[2][i], Mjoff_diff[3][i], Mjoff_diff[4][i], Mjoff_diff[5][i], Mjoff_diff[6][i], Mjoff_diff[7][i],
	   Kijoff_diff[0][i], Kijoff_diff[1][i], Kijoff_diff[2][i], Kijoff_diff[3][i], Kijoff_diff[4][i], Kijoff_diff[5][i], Kijoff_diff[6][i], Kijoff_diff[7][i]);
  }
  printf("\n");

  return 0;
}
