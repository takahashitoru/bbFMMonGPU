#include <stdio.h>
#include <stdlib.h>

/* based on aux_CPU9F.c */
/* based on aux_CPU9A.c */
/* same as aux_scuda45I.c */
/* based on aux_scuda38BH.c */

#define B 4
//#define B 2

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

  int Mjoff_diff[8][189];
  int Kijoff_diff[8][189];

  /* Loop over observation cell's sibling-index */
  for (int tz = 0; tz < 8; tz ++) {

    //    printf("#define B%d_COMPXYZ%d() ", B, tz);

    /* Compute the obs's sibling-index coordinates: 0<=lx,ly,lz<2 */
    int lx = (tz >> 2);       // tz/4
    int ly = ((tz & 3) >> 1); // (tz%4)/2
    int lz = (tz & 1);        // tz%2

    /* Initialise offsets */
    int Mjoff = 0;
    int Kijoff = 0;

    int n = 0;

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

	    int Mjoff_old = Mjoff;
	    int Kijoff_old = Kijoff;

	    Mjoff = (fdz * (2 * B + 4) + fdy) * (2 * B + 4) + fdx; // Mj[2*B+4][2*B+4][2*B+4]
	    Kijoff = Ktable[(kx + 3) + 7 * (ky + 3) + 49 * (kz + 3)];

	    //	    printf("COMP(%d, %d)", Kijoff - Kijoff_old, Mjoff - Mjoff_old);
	    //	    if (!(fdz == 5 && fdy == 5 && fdx == 5)) {
	    //	      printf("; ");
	    //	    }

	    Mjoff_diff[tz][n] = Mjoff - Mjoff_old;
	    Kijoff_diff[tz][n] = Kijoff - Kijoff_old;
	    n ++;

	  }
	}
      }
    }
    //    printf("\n");
  }
  

  printf("#define B%d_COMPXYZ() ", B);
  for (int n = 0; n < 189; n ++) {
    printf("COMP(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d)",
	   Kijoff_diff[0][n], Kijoff_diff[1][n], Kijoff_diff[2][n], Kijoff_diff[3][n], Kijoff_diff[4][n], Kijoff_diff[5][n], Kijoff_diff[6][n], Kijoff_diff[7][n],
	   Mjoff_diff[0][n], Mjoff_diff[1][n], Mjoff_diff[2][n], Mjoff_diff[3][n], Mjoff_diff[4][n], Mjoff_diff[5][n], Mjoff_diff[6][n], Mjoff_diff[7][n]);
    if (n < 188) {
      printf("; ");      
    } else {
      printf("\n");
    }
  }

  return 0;
}
