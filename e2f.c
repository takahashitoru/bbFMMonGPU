#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  float e = atof(argv[1]);
  printf("%20.8lf\n", e);
  return 0;
}
