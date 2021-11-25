#include"bbfmm.h"

void output(real3 *field, int Nf, int dof, real *phi)
{
#ifdef OUTPUT_FULL
  int i, j;
  for (i = 0; i < Nf; i++) {
    fprintf(stdout, "%d ", i);
    for (j = 0; j < dof; j++) {
      fprintf(stdout, "%24.16e ", phi[i * dof + j]);
    }
    fprintf(stdout, "%24.16e %24.16e %24.16e\n", field[i].x, field[i].y, field[i].z);
  }
#else
  fwrite(phi, sizeof(real), Nf * dof, stdout);
#endif
}
