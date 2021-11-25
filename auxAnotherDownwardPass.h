#ifndef AUXANOTHERDOWNWARDPASS_H
#define AUXANOTHERDOWNWARDPASS_H

#include "bbfmm.h"

EXTERN void set_field_points_and_sources(int nfield, void *field, void *field4, real *phi, int nsource, void *source, real *q, void *source4);

EXTERN void read_phi_from_field4(int nfield, void *field4, real *phi);

EXTERN void compute_Nfwork(int Asta, int Aend, cell *c, int *Nfwork);

EXTERN void compute_fieldlistwork(int Asta, int Aend, cell *c, int pitch_fieldlist, int *fieldlist, int *fieldlistwork);

EXTERN void create_sources(int maxlev, int *levsta, int *levend, cell *c, int *sourcelist, int *Nallsource, int *allsourcelist, int pitch_allsourcelist);

EXTERN void auxAnotherDownwardPass_sort_particles(int Asta, int Aend,
						  int *fieldlist, int *fieldsta, int *fieldend,
						  int *sourcelist, int *sourcesta, int *sourceend,
						  real3 *field, real *phi, real *q,
						  real3 *field_sorted, real *phi_sorted, real *q_sorted);

EXTERN void auxAnotherDownwardPass_sort_particles2(int Asta, int Aend,
						   int *fieldlist, int *fieldsta, int *fieldend,
						   int *sourcelist, int *sourcesta, int *sourceend,
						   real *phi, real3 *source, real *q,
						   real *field_sorted, real4 *source_sorted);

EXTERN void auxAnotherDownwardPass_read_sorted_field_values(int Asta, int Aend,
							    int *fieldlist, int *fieldsta, int *fieldend,
							    real *phi_sorted, real *phi);

#endif /* AUXANOTHERDOWNWARDPASS_H */
