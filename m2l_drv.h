#ifndef M2L_DRV_H
#define M2L_DRV_H

#define M2L_EXIT_SUCCESS 0
#define M2L_EXIT_FAIL    1

#define M2L_SCHEME_BA 0
#define M2L_SCHEME_SI 1
#define M2L_SCHEME_CL 2
#define M2L_SCHEME_IJ 3

#ifndef PITCH_SOURCECLUSTERS
#define PITCH_SOURCECLUSTERS 32 // 27 or more
#endif

#ifndef CHUNK_SIZE_CL
#define CHUNK_SIZE_CL 2 // Currently, only 2
#endif
#ifndef TILE_SIZE_ROW_CL
#define TILE_SIZE_ROW_CL 32 // Currently, only 32
#endif
#ifndef TILE_SIZE_COLUMN_CL
#define TILE_SIZE_COLUMN_CL 32 // Currently, only 32
#endif

#ifndef CHUNK_SIZE_IJ
#define CHUNK_SIZE_IJ 4 // Currently, only 4
#endif
#ifndef LEVEL_SWITCH_IJ
#define LEVEL_SWITCH_IJ 2
#endif
#ifndef NUM_ROW_GROUPS_IJ
//#define NUM_ROW_GROUPS_IJ 8 // good for C2050+SDK3.2
#define NUM_ROW_GROUPS_IJ 4
#endif

#include "m2l_host.h"

#include "m2l_host_basic.cu"
#include "m2l_host_sibling_blocking.cu"
#include "m2l_host_cluster_blocking.cu"
#include "m2l_host_ij_blocking.cu"

#endif /* M2L_DRV_H */
