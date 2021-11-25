#include "auxAnotherDownwardPass.h"

#define MINLOOP (128)

void set_field_points_and_sources(int nfield, void *field, void *field4, real *phi, int nsource, void *source, real *q, void *source4)
{
  real3 *Field = (real3 *)field; // cast
  real4 *Field4 = (real4 *)field4; // cast
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < nfield; i ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    Field4[i].x = Field[i].x;
    Field4[i].y = Field[i].y;
    Field4[i].z = Field[i].z;
    Field4[i].w = phi[i];
  }
  
  real3 *Source = (real3 *)source; // cast
  real4 *Source4 = (real4 *)source4; // cast
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < nsource; i ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    Source4[i].x = Source[i].x;
    Source4[i].y = Source[i].y;
    Source4[i].z = Source[i].z;
    Source4[i].w = q[i]; 
  }
}


void create_sources(int maxlev, int *levsta, int *levend, cell *c, int *sourcelist, int *Nallsource, int *allsourcelist, int pitch_allsourcelist)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    int n = 0; // number of sources in all the neighbour cells of A
    for (int ineighbor = 0; ineighbor < c->ineigh[A]; ineighbor ++) {
      int B = c->neighbors[c->pitch_neighbors * A + ineighbor];
#ifdef ENABLE_ORIGINAL_FIELD_SOURCE_LISTS
      for (int i = 0; i < c->Ns[B]; i ++) {
	allsourcelist[pitch_allsourcelist * (A - levsta[maxlev]) + n]
	  = sourcelist[c->pitch_sourcelist * B + i]; // c->sourcelist[c->pitch_sourcelist * B+i]
	n ++;
      }
#else
      for (int i = 0; i < c->sourceend[B] - c->sourcesta[B] + 1; i ++) {
	allsourcelist[pitch_allsourcelist * (A - levsta[maxlev]) + n]
	  = sourcelist[c->sourcesta[B] + i]; // (*atree)->sourcelist[c->sourcesta[B]+i]
	n ++;
      }
#endif
    }
    Nallsource[A - levsta[maxlev]] = n;
  }
}


void compute_Nfwork(int Asta, int Aend, cell *c, int *Nfwork)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = Asta; A <= Aend; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    Nfwork[A - Asta] = c->fieldend[A] - c->fieldsta[A] + 1;
  }
}


void compute_fieldlistwork(int Asta, int Aend, cell *c, int pitch_fieldlist, int *fieldlist, int *fieldlistwork)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = Asta; A <= Aend; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
#if(0)
    int n = c->fieldend[A] - c->fieldsta[A] + 1;
    int *fieldlistworkA = &(fieldlistwork[pitch_fieldlist * (A - Asta)]);
    int *fieldlistA = &(fieldlist[c->fieldsta[A]]);
    for (int i = 0; i < n; i ++) {
      fieldlistworkA[i] = fieldlistA[i];
    }
#else
    for (int i = 0; i < c->fieldend[A] - c->fieldsta[A] + 1; i ++) {
      fieldlistwork[pitch_fieldlist * (A - Asta) + i] = fieldlist[c->fieldsta[A] + i];
    }
#endif
  }
}

void read_phi_from_field4(int nfield, void *field4, real *phi)
{
  real4 *Field4 = (real4 *)field4; // cast
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < nfield; i ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    phi[i] = Field4[i].w;
  }
}


void auxAnotherDownwardPass_sort_particles(int Asta, int Aend,
					   int *fieldlist, int *fieldsta, int *fieldend,
					   int *sourcelist, int *sourcesta, int *sourceend,
					   real3 *field, real *phi, real *q,
					   real3 *field_sorted, real *phi_sorted, real *q_sorted)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = Asta; A <= Aend; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    for (int i = fieldsta[A]; i <= fieldend[A]; i ++) {
      int findex = fieldlist[i];
      field_sorted[i] = field[findex];
      phi_sorted[i] = phi[findex];
    }
    for (int j = sourcesta[A]; j <= sourceend[A]; j ++) { // source=field
      int sindex = sourcelist[j];
      q_sorted[j] = q[sindex];
    }
  }
}


void auxAnotherDownwardPass_sort_particles2(int Asta, int Aend,
					    int *fieldlist, int *fieldsta, int *fieldend,
					    int *sourcelist, int *sourcesta, int *sourceend,
					    real *phi, real3 *source, real *q,
					    real *field_sorted, real4 *source_sorted)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = Asta; A <= Aend; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    for (int i = fieldsta[A]; i <= fieldend[A]; i ++) {
      int findex = fieldlist[i];
      field_sorted[i] = phi[findex];
    }
    for (int j = sourcesta[A]; j <= sourceend[A]; j ++) { // source=field
      int sindex = sourcelist[j];
      source_sorted[j].x = source[sindex].x;
      source_sorted[j].y = source[sindex].y;
      source_sorted[j].z = source[sindex].z;
      source_sorted[j].w = q[sindex];
    }
  }
}


void auxAnotherDownwardPass_read_sorted_field_values(int Asta, int Aend,
						     int *fieldlist, int *fieldsta, int *fieldend,
						     real *phi_sorted, real *phi)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int A = Asta; A <= Aend; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    for (int i = fieldsta[A]; i <= fieldend[A]; i ++) {
      int findex = fieldlist[i];
      phi[findex] = phi_sorted[i];
    }
  }
}
