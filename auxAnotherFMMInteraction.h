#ifndef AUXANOTHERFMMINTERACTION_H
#define AUXANOTHERFMMINTERACTION_H

#include "bbfmm.h"

#ifndef POW2
#define POW2(n) (1 << (n))
#endif

EXTERN void aux_store_field_values(int dofn3, int pitch2, int Asta, int Aend, real *fieldval, real *F);

EXTERN void aux_convert_E_cluster_blocking(int cutoff, int *Ktable, real *E, real *K);

EXTERN void aux_convert_E_ij_blocking_real4(int cutoff, int *Ktable, real *E, void *Kanother);
EXTERN void aux_convert_E_ij_blocking_real2(int cutoff, int *Ktable, real *E, void *Kanother);
EXTERN void aux_convert_E_ij_blocking(int cutoff, int *Ktable, real *E, real *Kanother);

EXTERN void aux_convert_PS_ij_blocking(int cutoff, int *levsta, int *levend, real3 *center, real *celeng,
				       int switchlevel_high, int maxlev, int *PSanotherstart,
				       real *PS, real *PSanother);

EXTERN void aux_convert_PF_ij_blocking_real4(int cutoff, int *levsta, int *levend, real iL, real3 *center,
					       int level, int Dx, int Dy, int Dz, real *PFtmp, void *PF);
EXTERN void aux_convert_PF_ij_blocking_real2(int cutoff, int *levsta, int *levend, real iL, real3 *center,
					       int level, int Dx, int Dy, int Dz, real *PFtmp, void *PF);
EXTERN void aux_convert_PF_ij_blocking(int cutoff, int *levsta, int *levend, real iL, real3 *center,
				       int level, int Dx, int Dy, int Dz, real *PFtmp, real *PF);


EXTERN void postm2l_convert_U_from_column_major_to_row_major(real *Ucol, real *Urow, int dofn3, int cutoff);
EXTERN void postm2l_compute_adjusting_vector(real *Kweights, real *adjust, int dof, int n3);
EXTERN void postm2l(int Asta, int Aend, int dofn3, int cutoff, real *Urow, real *PF, real scale, real *adjust, real *F);

#endif /* AUXANOTHERFMMINTERACTION_H */
