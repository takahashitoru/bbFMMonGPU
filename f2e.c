#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  float e = atof(argv[1]);
  printf("%14.7e\n", e);
  return 0;
}
