#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main(int argc, char **argv)
{
  FILE *fpdir, *fpfmm;
  int idummy, N;
  double phidir, phifmm, x1, x2, x3, err, tmp, sumdir, sumfmm;

  if(argc<3){
    //    fprintf(stderr, "usage: %s (dir-output) (fmm-output)\n", argv[0]);  
    fprintf(stderr, "usage: %s (output) (another output)\n", argv[0]);  
    exit(1);
  }else{
    //    fprintf(stderr, "dir-output: %s\n", argv[1]);
    //    fprintf(stderr, "fmm-output: %s\n", argv[2]);
    fprintf(stderr, "output: %s\n", argv[1]); // usually, DIRECT
    fprintf(stderr, "another output: %s\n", argv[2]); // usually, FMM
  }

  fpdir=fopen(argv[1], "r");
  fpfmm=fopen(argv[2], "r");

  err=0;
  sumdir=0;
  sumfmm=0;
  while(fscanf(fpdir, "%d %lf %lf %lf %lf", &idummy, &phidir, &x1, &x2, &x3)==5
	&& fscanf(fpfmm, "%d %lf %lf %lf %lf", &idummy, &phifmm, &x1, &x2, &x3)==5){
    tmp=phidir-phifmm;
    err+=tmp*tmp;
    sumdir+=phidir*phidir;
    sumfmm+=phifmm*phifmm;
  }    
  err=sqrt(err);
  sumdir=sqrt(sumdir);
  sumfmm=sqrt(sumfmm);
  N=idummy+1;

  fprintf(stdout, "%d %18.12e %18.12e %18.12e %18.12e\n", N, err/sumdir, sumdir, sumfmm, err);

  fclose(fpdir);
  fclose(fpfmm);

  return 0;
}
