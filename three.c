#include<stdio.h>
int main()
{
  /* -16<=X,Y,Z,A,B,C<16 */
  int X =  8;
  int Y = -8;
  int Z = -11;
  int A =  6;
  int B = -4;
  int C = 12;
  unsigned int XYZ, ABC, diff;
  //  XYZ = ((X + 16) << 12) + ((Y + 16) << 6) + (Z + 16);
  //  ABC = ((A + 16) << 12) + ((B + 16) << 6) + (C + 16);
  XYZ = (X + 16) * 64 * 64 + (Y + 16) * 64 + (Z + 16);
  ABC = (A + 16) * 64 * 64 + (B + 16) * 64 + (C + 16);
  diff = XYZ - ABC;
  //  printf("%d %d %d\n", (diff >> 12) - 32, (diff >> 6) - 32, (diff & 63) - 32);
  printf("%d %d %d\n", diff / 64 / 64 - 32, diff / 64 - 32, diff % 64 - 32);
  return 0;
}
