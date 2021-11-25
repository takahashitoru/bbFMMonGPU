#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  int nlines = atoi(argv[1]);
  int ntrials = atoi(argv[2]);
  FILE *fp = fopen(argv[3], "r");
  FILE *fp2 = fopen(argv[4], "r");

  if (strcmp(argv[5], "calc_performance")) {

    int i, j;
    for (i = 0; i < nlines; i += ntrials) {
      double sum = 0.0, x;
      for (j = 0; j < ntrials; j ++) {
	fscanf(fp, "%lf", &x);
	sum += x;
      }
      char item[30];
      fscanf(fp2, "%s", item);
      fprintf(stdout, "# %s: %s = %14.7e\n", argv[5], item, sum / ntrials);
    }

  } else {

    int i, j;
    for (i = 0; i < nlines; i += ntrials) {
      double sum = 0.0, x;
      for (j = 0; j < ntrials; j ++) {
	fscanf(fp, "%lf", &x);
	sum += x;
      }
      char item[30];
      fscanf(fp2, "%s", item);
      if (sum != 0.0) {
	fprintf(stdout, "# %s: %s = %f [Gflop/s]\n", argv[5], item, sum / ntrials);
      } else {
	fprintf(stdout, "# %s: %s = (sec is zero) [Gflop/s]\n", argv[5], item, sum / ntrials);
      }
    }

  }
    
  fclose(fp);
  fclose(fp2);
}
