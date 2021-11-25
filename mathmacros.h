#ifndef MATHMACROS_H
#define MATHMACROS_H

#define FLOORUP(n, m) (((n) & ((m) - 1)) ? ((((n) / (m)) + 1) * (m)) : (n))

#ifndef MAX
#define MAX(x, y) (( (x) > (y) ) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x, y) (( (x) < (y) ) ? (x) : (y))
#endif

#ifndef POW8
#define POW8(i) (1 << (3 * (i)))
#endif

#ifndef DIV8
#define DIV8(i) ( (i) >> 3 )
#endif

#ifndef MOD8
#define MOD8(i) ( (i) & 7 )
#endif

#ifndef POW2
#define POW2(i) ( 1 << (i) )
#endif

#ifndef CUBE
#define CUBE(i) ( (i) * (i) * (i) )
#endif

#endif /* MATHMACROS_H */

