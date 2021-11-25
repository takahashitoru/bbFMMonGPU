#include <stdio.h>
#include <stdlib.h>

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

  /* Loop over observation cell's sibling-index */
  for (int tz = 0; tz < 8; tz ++) {

    //110312    printf("#define COMPXYZ%d() ", tz);
    printf("#define B%d_COMPXYZ%d() ", B, tz);

    /* Compute the obs's sibling-index coordinates: 0<=lx,ly,lz<2 */
    int lx = (tz >> 2);       // tz/4
    int ly = ((tz & 3) >> 1); // (tz%4)/2
    int lz = (tz & 1);        // tz%2

    /* Initialise offsets */
    int Mjoff = 0;
    int Kijoff = 0;

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

	    //120214	    /* Obtain the source's coordinates in (B+2)x(B+2)x(B+2)
	    //120214	       region (except for the offset due to obs's
	    //120214	       sibling-index) */
	    //120214	    int ix = fdx / 2; // 0<=ix<(B+2)/2
	    //120214	    int iy = fdy / 2; // 0<=iy<(B+2)/2
	    //120214	    int iz = fdz / 2; // 0<=iz<(B+2)/2

	    int Mjoff_old = Mjoff;
	    int Kijoff_old = Kijoff;
	    Mjoff = ((sib * (B + 2) + iz) * (B + 2) + iy) * (B + 2) + ix; // Mj[8][B+2][B+2][B+2]
	    Kijoff = Ktable[(kx + 3) + 7 * (ky + 3) + 49 * (kz + 3)];

	    printf("COMP(%d, %d)", Kijoff - Kijoff_old, Mjoff - Mjoff_old);
	    if (!(fdz == 5 && fdy == 5 && fdx == 5)) {
	      printf("; ");
	    }

	  }
	}
      }
    }
    printf("\n");
  }

  return 0;
}
