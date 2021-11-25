#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[])
{
  double a, b, c, d;
  sscanf(argv[1], "%lf", &a);
  sscanf(argv[2], "%lf", &b);
  sscanf(argv[3], "%lf", &c);
  sscanf(argv[4], "%lf", &d);
  printf("%15.7e\n", a - b - c - d);
  return 0;
}
