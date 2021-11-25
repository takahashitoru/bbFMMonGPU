#include "bbfmm.h"

static void infoCell(const cell *c, const int A)
{
  DBG("A=%d\n", A);
  DBG("parent=%d\n", GET_PARENT_INDEX(A));
  MSG("leaves: ");
  for (int i = 0; i < 8; i++) {
    fprintf(stderr, "%d ", GET_CHILD_INDEX(A, i));
  }
  fprintf(stderr, "\n");
  
  DBG("ineigh=%d: ", c->ineigh[A]);
  for (int i = 0; i < c->ineigh[A]; i++) {
    fprintf(stderr, "%d ", c->neighbors[c->pitch_neighbors * A + i]);
  }
  fprintf(stderr, "\n");

  DBG("iinter=%d: ", c->iinter[A]);
  for (int i = 0; i < c->iinter[A]; i++) {
    fprintf(stderr, "%d ", c->interaction[c->pitch_interaction * A + i]);
  }
  fprintf(stderr, "\n");
}  

#define plotPoint(x, y, z) (fprintf(fp, "%24.15e %24.15e %24.15e \n", x, y, z))

static  void drawCell(FILE *fp, FILE *fp2, real3 c, real length, int index)
{
  real half = length / 2;
  
  plotPoint(c.x + half, c.y + half, c.z + half);
  plotPoint(c.x - half, c.y + half, c.z + half);
  plotPoint(c.x - half, c.y - half, c.z + half);
  plotPoint(c.x + half, c.y - half, c.z + half);
  plotPoint(c.x + half, c.y + half, c.z + half);
  fprintf(fp, "\n"); /* brank line */
  plotPoint(c.x + half, c.y + half, c.z - half);
  plotPoint(c.x - half, c.y + half, c.z - half);
  plotPoint(c.x - half, c.y - half, c.z - half);
  plotPoint(c.x + half, c.y - half, c.z - half);
  plotPoint(c.x + half, c.y + half, c.z - half);
  fprintf(fp, "\n"); /* brank line */
  plotPoint(c.x + half, c.y + half, c.z + half);
  plotPoint(c.x + half, c.y + half, c.z - half);
  fprintf(fp, "\n"); /* brank line */
  plotPoint(c.x - half, c.y + half, c.z + half);
  plotPoint(c.x - half, c.y + half, c.z - half);
  fprintf(fp, "\n"); /* brank line */
  plotPoint(c.x - half, c.y - half, c.z + half);
  plotPoint(c.x - half, c.y - half, c.z - half);
  fprintf(fp, "\n"); /* brank line */
  plotPoint(c.x + half, c.y - half, c.z + half);
  plotPoint(c.x + half, c.y - half, c.z - half);
  fprintf(fp, "\n"); /* brank line */

  fprintf(fp2, "set label \"%d\" at %24.15e, %24.15e, %24.15e\n", index, c.x, c.y, c.z);
}

#define CHECKTREE_TMPFILE "foofoo"
#define CHECKTREE_TMPFILE2 "foofoo2"

void checkTree(anotherTree *atree)
{
  const int maxlev = atree->maxlev;
  DBG("maxlev = %d \n", maxlev);
  const int *levsta = atree->levsta;
  const int *levend = atree->levend;
  const real *celeng = atree->celeng;
  const cell *c = atree->c;

  FILE *fp = fopen(CHECKTREE_TMPFILE, "w");
  FILE *fp2 = fopen(CHECKTREE_TMPFILE2, "w");

  for (int l = 0; l <= maxlev; l ++) {
    DBG("l=%d levsta=%d levend=%d celeng=%15.7e \n", l, levsta[l], levend[l], celeng[l]);
    for (int A = levsta[l]; A <= levend[l]; A ++) {
      drawCell(fp, fp2, c->center[A], celeng[l], A);
#if(1)
      infoCell(c, A);
#endif
    }
  }     
  fclose(fp);
  fclose(fp2);
}

#define PLOTPOINTS_TMPFILE "barbar"
#define PLOTPOINTS_TMPFILE2 "barbar2"

void checkPlotPoints(int n, real3 *point)
{
  int i;
  FILE *fp, *fp2;

  fp=fopen(PLOTPOINTS_TMPFILE, "w");
  fp2=fopen(PLOTPOINTS_TMPFILE2, "w");

  for (i = 0; i < n; i++) {
    fprintf(fp, "%24.15e %24.15e %24.15e\n", point[i].x, point[i].y, point[i].z);
    fprintf(fp2, "set label \"%d\" at %24.15e, %24.15e, %24.15e\n", i, point[i].x, point[i].y, point[i].z);
  }

  fclose(fp);
  fclose(fp2);
}

void calc_performance(char *str, double flop, double sec)
{
  double perf;
  if (sec > 0) {
    perf = (flop / giga) / sec; /* [Gflop/s] */
    INFO("%s = %f [Gflop/s]\n", str, perf); /* Be care of overflow */
  } else {
    INFO("%s = (sec is zero) [Gflop/s]\n", str);
  }
}


void estimate_performacne_bounded_by_bandwidth(char *str, double size_load_store_bytes,
					       double peak_bandwidth_giga_bytes, double flop)
{
  double access_time, perf;
  access_time = size_load_store_bytes / peak_bandwidth_giga_bytes; /* [s/G] */
  perf = flop / access_time; /* [Gflop/s] */
  INFO("%s = %f [Gflop/s]\n", str, perf);
}
