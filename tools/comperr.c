#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define NMAX (100000000)
#define real float

int main(int argc, char **argv)
{
  FILE *fpdir, *fpfmm;
  int idummy, N, i;
  real phifmm[NMAX];
  double phidir[NMAX], dphifmm, x1, x2, x3, err, tmp, sumdir, sumfmm;
  
  if (argc < 3) {
    fprintf(stderr, "usage: %s (output) (another output)\n", argv[0]);  
    exit(1);
  } else {
    fprintf(stderr, "output: %s\n", argv[1]); // usually, DIRECT
    fprintf(stderr, "another output: %s\n", argv[2]); // usually, FMM
  }

  /* Read the output of DIRECT calculation */
  fpdir = fopen(argv[1], "r");
  if (fpdir == NULL) {
    fprintf(stderr, "Fail to open %s.\n", argv[1]);
    exit(1);
  } 

  N = 0;
  while(fscanf(fpdir, "%d %lf %lf %lf %lf", &idummy, phidir + N, &x1, &x2, &x3) == 5) {
    N++;
    if (N > NMAX) {
      fprintf(stderr, "NMAX is too small.\n");
      exit(1);
    }
  }    
  fprintf(stderr, "Number of particles for DIRECT: N=%d\n", N);
  fclose(fpdir);

  /* Read the output of FMM calculation */
  fpfmm = fopen(argv[2], "r");
  if (fpfmm == NULL) {
    fprintf(stderr, "Fail to open %s.\n", argv[2]);
    exit(1);
  } 
  fread(phifmm, sizeof(real), N, fpfmm);
  fclose(fpfmm);

  /* Compute the error */
  err = 0;
  sumdir = 0;
  sumfmm = 0;
  for (i = 0; i < N; i++) {
    dphifmm = (double)phifmm[i];
    tmp = phidir[i] - dphifmm;
    err += tmp * tmp;
    sumdir += phidir[i] * phidir[i];
    sumfmm += dphifmm * dphifmm;
  }    
  err = sqrt(err);
  sumdir = sqrt(sumdir);
  sumfmm = sqrt(sumfmm);
  
  fprintf(stdout, "%d %18.12e %18.12e %18.12e %18.12e\n", N, err / sumdir, sumdir, sumfmm, err);
  
  return 0;
}
