#include <stdio.h>
#include <stdlib.h>

/* based on aux_CPU9N.c */
/* based on aux_CPU9F.c */
/* based on aux_CPU9A.c */
/* same as aux_scuda45I.c */
/* based on aux_scuda38BH.c */

//#define B 2
//#define B 4
#define B 8

//#define NINTER 1
//#define NINTER 2
//#define NINTER 4
//#define NINTER 8
#define NINTER 16

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

    //    printf("#define B%d_COMPXYZ%d() ", B, tz);
    printf("#define COMPXYZ_B%d_I%d_S%d() ", B, NINTER, tz);

    /* Compute the obs's sibling-index coordinates: 0<=lx,ly,lz<2 */
    int lx = (tz >> 2);       // tz/4
    int ly = ((tz & 3) >> 1); // (tz%4)/2
    int lz = (tz & 1);        // tz%2

    //    /* Initialise offsets */
    //    int Mjoff = 0;
    //    int Kijoff = 0;

    int Mjoff[189];
    int Kijoff[189];
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

	    //	    int Mjoff_old = Mjoff;
	    //	    int Kijoff_old = Kijoff;
	    //
	    //	    Mjoff = (fdz * (2 * B + 4) + fdy) * (2 * B + 4) + fdx; // Mj[2*B+4][2*B+4][2*B+4]
	    //	    Kijoff = Ktable[(kx + 3) + 7 * (ky + 3) + 49 * (kz + 3)];
	    //
	    //	    printf("CMP%d(%d, %d)", B, Kijoff - Kijoff_old, Mjoff - Mjoff_old);
	    //	    if (!(fdz == 5 && fdy == 5 && fdx == 5)) {
	    //	      printf("; ");
	    //	    }

	    Mjoff[n] = (fdz * (2 * B + 4) + fdy) * (2 * B + 4) + fdx; // Mj[2*B+4][2*B+4][2*B+4]
	    Kijoff[n] = Ktable[(kx + 3) + 7 * (ky + 3) + 49 * (kz + 3)];
	    n ++;

	  }
	}
      }
    }
    //    printf("\n");

#if (NINTER == 1)    
    for (int i = 0; i < 188; i ++) {
      printf("CMP%d%d(%d,%d); ", B, 1, Kijoff[i], Mjoff[i]);
    }
    printf("CMP%d%d(%d,%d)\n", B, 1, Kijoff[188], Mjoff[188]);
#elif (NINTER == 2)    
    for (int i = 0; i < 188; i += 2) { // (0,1)(2,3)...(186,187)
      printf("CMP%d%d(%d,%d,%d,%d); ", B, 2, Kijoff[i], Kijoff[i + 1], Mjoff[i], Mjoff[i + 1]);
    }
    printf("CMP%d%d(%d,%d)\n", B, 1, Kijoff[188], Mjoff[188]);
#elif (NINTER == 4)
    for (int i = 0; i < 188; i += 4) { // (0,1,2,3)...(184,185,186,187)
      printf("CMP%d%d(%d,%d,%d,%d,%d,%d,%d,%d); ", B, 4, Kijoff[i], Kijoff[i + 1], Kijoff[i + 2], Kijoff[i + 3], Mjoff[i], Mjoff[i + 1], Mjoff[i + 2], Mjoff[i + 3]);
    }
    printf("CMP%d%d(%d,%d)\n", B, 1, Kijoff[188], Mjoff[188]);
#elif (NINTER == 8)
    for (int i = 0; i < 184; i += 8) { // (0,1,2,3,4,5,6,7)...(176,177,178,179,180,181,182,183)
      printf("CMP%d%d(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d); ", B, 8, Kijoff[i], Kijoff[i + 1], Kijoff[i + 2], Kijoff[i + 3], Kijoff[i + 4], Kijoff[i + 5], Kijoff[i + 6], Kijoff[i + 7], Mjoff[i], Mjoff[i + 1], Mjoff[i + 2], Mjoff[i + 3], Mjoff[i + 4], Mjoff[i + 5], Mjoff[i + 6], Mjoff[i + 7]);
    }
    printf("CMP%d%d(%d,%d,%d,%d,%d,%d,%d,%d); ", B, 4, Kijoff[184], Kijoff[185], Kijoff[186], Kijoff[187], Mjoff[184], Mjoff[185], Mjoff[186], Mjoff[187]);
    printf("CMP%d%d(%d,%d)\n", B, 1, Kijoff[188], Mjoff[188]);
#elif (NINTER == 16)
    for (int i = 0; i < 176; i += 16) { // (0..15)...(170..175)
      printf("CMP%d%d(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d); ", B, 16, Kijoff[i], Kijoff[i + 1], Kijoff[i + 2], Kijoff[i + 3], Kijoff[i + 4], Kijoff[i + 5], Kijoff[i + 6], Kijoff[i + 7], Kijoff[i + 8], Kijoff[i + 9], Kijoff[i + 10], Kijoff[i + 11], Kijoff[i + 12], Kijoff[i + 13], Kijoff[i + 14], Kijoff[i + 15], Mjoff[i], Mjoff[i + 1], Mjoff[i + 2], Mjoff[i + 3], Mjoff[i + 4], Mjoff[i + 5], Mjoff[i + 6], Mjoff[i + 7], Mjoff[i + 8], Mjoff[i + 9], Mjoff[i + 10], Mjoff[i + 11], Mjoff[i + 12], Mjoff[i + 13], Mjoff[i + 14], Mjoff[i + 15]);
    }
    printf("CMP%d%d(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d); ", B, 8, Kijoff[176], Kijoff[177], Kijoff[178], Kijoff[179], Kijoff[180], Kijoff[181], Kijoff[182], Kijoff[183], Mjoff[176], Mjoff[177], Mjoff[178], Mjoff[179], Mjoff[180], Mjoff[181], Mjoff[182], Mjoff[183]);
    printf("CMP%d%d(%d,%d,%d,%d,%d,%d,%d,%d); ", B, 4, Kijoff[184], Kijoff[185], Kijoff[186], Kijoff[187], Mjoff[184], Mjoff[185], Mjoff[186], Mjoff[187]);
    printf("CMP%d%d(%d,%d)\n", B, 1, Kijoff[188], Mjoff[188]);
#else
#error NINTER is not defined
#endif

  } // tz

  return 0;
}
