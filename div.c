#include<stdio.h>
#include<math.h>

int main(int argc, char *argv[])
{
  float f, g;
  sscanf(argv[1], "%f", &f);
  sscanf(argv[2], "%f", &g);
  printf("%14.7e\n", f / g);
  return 0;
}
