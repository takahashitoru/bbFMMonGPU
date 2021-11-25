#include<stdio.h>
#include<math.h>

int main(int argc, char *argv[])
{
  int n;
  float f;
  sscanf(argv[1], "%f", &f);
  sscanf(argv[2], "%d", &n);
  switch (n) {
  case 3:
#if defined(ENABLE_F_FORMAT)
    fprintf(stdout, "%8.3f\n", f);
#else
    fprintf(stdout, "%10.3e\n", f);
#endif
    break;
  case 2:
#if defined(ENABLE_F_FORMAT)
    fprintf(stdout, "%8.2f\n", f);
#else
    fprintf(stdout, "%9.2e\n", f);
#endif
    break;
  case 1:
#if defined(ENABLE_F_FORMAT)
    fprintf(stdout, "%8.1f\n", f);
#else
    fprintf(stdout, "%8.1e\n", f);
#endif
    break;
  default:
    fprintf(stderr, "invaild\n");
    return 1;
  }
  return 0;
}
