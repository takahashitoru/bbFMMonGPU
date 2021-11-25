#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#if defined(SINGLE)
#define real float
#define SQRT(x) sqrtf(x)
#define ZERO (0.0f)
#else
#define real double
#define SQRT(x) sqrt(x)
#define ZERO (0.0)
#endif

void read_binary(char *filename, int N, real *phi)
{
  FILE *fp;
  fp = fopen(filename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Fail to open %s.\n", filename);
    exit(EXIT_FAILURE);
  } 
  
  if (fread(phi, sizeof(real), N, fp) < N) { // dof=1 is supposed
    fprintf(stderr, "Fail to read %s.\n", filename);
    exit(EXIT_FAILURE);
  }
  fclose(fp);
}

int main(int argc, char **argv)
{
  FILE *fpdir, *fpfmm;
  int idummy, N, i;
  //  real *phi0, *phi1, err, tmp, sum0, sum1;
  
  if (argc < 4) {
#if defined(SINGLE)
    fprintf(stderr, "usage: %s (N) (output) (another output) SINGLE\n", argv[0]);
#else
    fprintf(stderr, "usage: %s (N) (output) (another output) DOUBLE\n", argv[0]);
#endif
    exit(1);
  } else {
    N = atoi(argv[1]);
    //////////////////////////////////////////////////////////////////////////
#if(0)
    fprintf(stderr, "N=%d\n", N);
    fprintf(stderr, "output=%s\n", argv[2]);
    fprintf(stderr, "another output=%s\n", argv[3]);
#endif
    //////////////////////////////////////////////////////////////////////////
  }

  real *phi0 = (real *)malloc(N * sizeof(real));
  real *phi1 = (real *)malloc(N * sizeof(real));

  read_binary(argv[2], N, phi0);
  read_binary(argv[3], N, phi1);

#if defined(SUM_IN_DOUBLE)
  double err = 0.0;
  double sum0 = 0.0;
  double sum1 = 0.0;
  double tmp;
#else
  real err = ZERO;
  real sum0 = ZERO;
  real sum1 = ZERO;
  real tmp;
#endif
  for (i = 0; i < N; i ++) {
    //////////////////////////////////////////////////////////////////////////
#if(0)
    fprintf(stderr, "i=%d phi0[i]=%15.7e phi1[i]=%15.7e\n", i, phi0[i], phi1[i]);
#endif
    //////////////////////////////////////////////////////////////////////////
    tmp = phi0[i] - phi1[i];
    err += tmp * tmp;
    sum0 += phi0[i] * phi0[i];
    sum1 += phi1[i] * phi1[i];
  }    
#if defined(SUM_IN_DOUBLE)
  err = sqrt(err);
  sum0 = sqrt(sum0);
  sum1 = sqrt(sum1);
#else
  err = SQRT(err);
  sum0 = SQRT(sum0);
  sum1 = SQRT(sum1);
#endif
  
  fprintf(stdout, "%d %18.12e %18.12e %18.12e %18.12e\n", N, err / sum0, sum0, sum1, err);
  
  free(phi0);
  free(phi1);

  return(EXIT_SUCCESS);
}
