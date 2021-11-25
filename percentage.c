#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[])
{
  float f, g, p;
  sscanf(argv[1], "%f", &f);
  sscanf(argv[2], "%f", &g);
  p = f / g * 100;
#if defined(DIGIT_ONE)
  if (p >= 100.0) {
    printf("%3.1f\n", p);
  } else if (p >= 10.0) {
    printf(" %2.1f\n", p);
  } else {
    printf("  %1.1f\n", p);
  }
#else
  if (p >= 100.0) {
    printf("%3.2f\n", p);
  } else if (p >= 10.0) {
    printf(" %2.2f\n", p);
  } else {
    printf("  %1.2f\n", p);
  }
#endif
  return 0;
}
