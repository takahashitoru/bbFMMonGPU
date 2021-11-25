#ifndef ANOTHERDOWNWARDPASS_CU
#define ANOTHERDOWNWARDPASS_CU

#include "bbfmm.h"
#include "eventTimer.h"

/**************************************************************************/
#if defined(CUDA_VER46)
/**************************************************************************/
#include "direct.cu"

#if !defined(DIRECT_GROUP_SIZE)
#define DIRECT_GROUP_SIZE (65535)
#endif

#if defined(FAST_HOST_CODE)
#include "auxAnotherDownwardPass.h"
#endif

#if !defined(DIRECT_NUM_THREADS_PER_BLOCK)
#define DIRECT_NUM_THREADS_PER_BLOCK (64)
#endif

#if (DOF != 1)
#error DOF=1 must be assumed.
#endif

#if defined(CUDA_VER46C) || defined(CUDA_VER46D) || defined(CUDA_VER46E) || defined(CUDA_VER46F) || defined(CUDA_VER46G) || defined(CUDA_VER46H) || defined(CUDA_VER46I) || defined(CUDA_VER46J) || defined(CUDA_VER46K) || defined(CUDA_VER46L) || defined(CUDA_VER46M) || defined(CUDA_VER46N) || defined(CUDA_VER46P) || defined(CUDA_VER46Q)
#if !defined(FIELDPOINTS_EQ_SOURCES)
#error FIELDPOINTS_EQ_SOURCES (Nf=Ns) must be assumed.
#endif
#endif

#if defined(ENABLE_NEARFIELD_BY_CPU)
extern "C" {
  void anotherDownwardPassNearField(anotherTree **atree, real3 *field, real3 *source, real *phi, real *q);
}
#endif


void anotherDownwardPass(anotherTree **atree, real3 *field, real3 *source,
			 real *Cweights, real *Tkz, real *q, real *U,
			 int cutoff, int n, int dof, real homogen, real *phi)
{
  /*
    Evaluation of far field interactions; L2L operation and evaluation by L
  */
  
  anotherDownwardPassX(atree, field, Cweights, Tkz, q, U, cutoff, n, dof, homogen, phi);

  /*
    Evaluation of near field interactions or direct computation;
    Assume that direct compution is done only at the maximum level
  */

  eventTimerType time_direct_all;
  initEventTimer(&time_direct_all);
  startEventTimer(&time_direct_all);

#if defined(ENABLE_NEARFIELD_BY_CPU) // specify any CPU version

  anotherNearField(atree, dof, homogen, field, source, phi, q);

#else

  eventTimerType time_direct_set, time_direct_kernel, time_direct_get;
  initEventTimer(&time_direct_set);
  initEventTimer(&time_direct_kernel);
  initEventTimer(&time_direct_get);

  /* Aliases */
  int ncell = (*atree)->ncell;
  int maxlev = (*atree)->maxlev;
  int *levsta = (*atree)->levsta;
  int *levend = (*atree)->levend;
  cell *c = (*atree)->c;
  int Nf = (*atree)->Nf; // number of all field points
  int Ns = (*atree)->Ns; // number of all sources
  int *ineigh = c->ineigh;
  int *neighbors = c->neighbors;
  int pitch_neighbors = c->pitch_neighbors;
  int *fieldlist = (*atree)->fieldlist;
  int *fieldsta = c->fieldsta;
  int *fieldend = c->fieldend;
#if defined(CUDA_VER46B) || defined(CUDA_VER46E) || defined(CUDA_VER46F) || defined(CUDA_VER46G) || defined(CUDA_VER46H) || defined(CUDA_VER46I) || defined(CUDA_VER46J) || defined(CUDA_VER46K) || defined(CUDA_VER46L) || defined(CUDA_VER46M) || defined(CUDA_VER46N) || defined(CUDA_VER46P) || defined(CUDA_VER46Q)
  int *sourcelist = (*atree)->sourcelist;
  int *sourcesta = c->sourcesta;
  int *sourceend = c->sourceend;
#endif

  DBG("ncell=%d Nf=%d Ns=%d\n", ncell, Nf, Ns);

  /* Allocate memory spaces on device */
  int *d_ineigh, *d_neighbors;
  CSC(cudaMalloc((void **)&d_ineigh, ncell * sizeof(int)));
  CSC(cudaMalloc((void **)&d_neighbors, pitch_neighbors * ncell * sizeof(int)));
  int *d_fieldsta, *d_fieldend;
  CSC(cudaMalloc((void **)&d_fieldsta, ncell * sizeof(int)));
  CSC(cudaMalloc((void **)&d_fieldend, ncell * sizeof(int)));
#if defined(CUDA_VER46F) || defined(CUDA_VER46G) || defined(CUDA_VER46H) || defined(CUDA_VER46I) || defined(CUDA_VER46J) || defined(CUDA_VER46K) || defined(CUDA_VER46L) || defined(CUDA_VER46M) || defined(CUDA_VER46N) || defined(CUDA_VER46P) || defined(CUDA_VER46Q)
  real *d_field_sorted;
  CSC(cudaMalloc((void **)&d_field_sorted, Nf * sizeof(real)));
  real4 *d_source_sorted;
  CSC(cudaMalloc((void **)&d_source_sorted, Ns * sizeof(real4)));
#elif defined(CUDA_VER46E)
  real3 *d_field_sorted;
  CSC(cudaMalloc((void **)&d_field_sorted, Nf * sizeof(real3)));
  real *d_phi_sorted, *d_q_sorted;
  CSC(cudaMalloc((void **)&d_phi_sorted, Nf * sizeof(real)));
  CSC(cudaMalloc((void **)&d_q_sorted, Ns * sizeof(real))); // Ns=Nf
#elif defined(CUDA_VER46C) || defined(CUDA_VER46D)
  int *d_fieldlist;
  CSC(cudaMalloc((void **)&d_fieldlist, Nf * sizeof(int)));
  real3 *d_field;
  CSC(cudaMalloc((void **)&d_field, Nf * sizeof(real3)));
  real *d_phi, *d_q;
  CSC(cudaMalloc((void **)&d_phi, Nf * sizeof(real)));
  CSC(cudaMalloc((void **)&d_q, Ns * sizeof(real))); // Ns=Nf
#elif defined(CUDA_VER46B)
  int *d_fieldlist;
  CSC(cudaMalloc((void **)&d_fieldlist, Nf * sizeof(int)));
  int *d_sourcelist, *d_sourcesta, *d_sourceend;
  CSC(cudaMalloc((void **)&d_sourcelist, Ns * sizeof(int)));
  CSC(cudaMalloc((void **)&d_sourcesta, ncell * sizeof(int)));
  CSC(cudaMalloc((void **)&d_sourceend, ncell * sizeof(int)));
#endif

  /* Set up field points and sources */
#if defined(CUDA_VER46F) || defined(CUDA_VER46G) || defined(CUDA_VER46H) || defined(CUDA_VER46I) || defined(CUDA_VER46J) || defined(CUDA_VER46K) || defined(CUDA_VER46L) || defined(CUDA_VER46M) || defined(CUDA_VER46N) || defined(CUDA_VER46P) || defined(CUDA_VER46Q)
  real *field_sorted = (real *)malloc(Nf * sizeof(real));
  real4 *source_sorted = (real4 *)malloc(Ns * sizeof(real4));
#if defined(FAST_HOST_CODE)
  auxAnotherDownwardPass_sort_particles2(levsta[maxlev], levend[maxlev],
					 fieldlist, fieldsta, fieldend,
					 sourcelist, sourcesta, sourceend,
					 phi, source, q,
					 field_sorted, source_sorted);
#else
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) {
    for (int i = fieldsta[A]; i <= fieldend[A]; i ++) {
      int findex = fieldlist[i];
      field_sorted[i] = phi[findex];
    }
    for (int j = sourcesta[A]; j <= sourceend[A]; j ++) { // source=field
      int sindex = sourcelist[j];
      source_sorted[j] = make_real4(source[sindex].x, source[sindex].y, source[sindex].z, q[sindex]);
    }
  }
#endif
#elif defined(CUDA_VER46E)
  real3 *field_sorted = (real3 *)malloc(Nf * sizeof(real3));
  real *phi_sorted = (real *)malloc(Nf * sizeof(real));
  real *q_sorted = (real *)malloc(Ns * sizeof(real));
#if defined(FAST_HOST_CODE)
  auxAnotherDownwardPass_sort_particles(levsta[maxlev], levend[maxlev],
					fieldlist, fieldsta, fieldend,
					sourcelist, sourcesta, sourceend,
					field, phi, q,
					field_sorted, phi_sorted, q_sorted);
#else
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) {
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
#endif
#elif defined(CUDA_VER46B)
  real4 *field4 = (real4 *)malloc(Nf * sizeof(real4));
  for (int i = 0; i < Nf; i ++) {
    field4[i] = make_real4(field[i].x, field[i].y, field[i].z, phi[i]); 
  }
  real4 *d_field4;
  CSC(cudaMalloc((void **)&d_field4, Nf * sizeof(real4)));

  real4 *source4 = (real4 *)malloc(Ns * sizeof(real4));
  for (int i = 0; i < Ns; i ++) {
    source4[i] = make_real4(source[i].x, source[i].y, source[i].z, q[i]); 
  }
  real4 *d_source4;
  CSC(cudaMalloc((void **)&d_source4, Ns * sizeof(real4)));
#endif

  /* Copy data to device */
  startEventTimer(&time_direct_set);
  CSC(cudaMemcpy(d_ineigh, ineigh, ncell * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_neighbors, neighbors, pitch_neighbors * ncell * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_fieldsta, fieldsta, ncell * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_fieldend, fieldend, ncell * sizeof(int), cudaMemcpyHostToDevice));
#if defined(CUDA_VER46F) || defined(CUDA_VER46G) || defined(CUDA_VER46H) || defined(CUDA_VER46I) || defined(CUDA_VER46J) || defined(CUDA_VER46K) || defined(CUDA_VER46L) || defined(CUDA_VER46M) || defined(CUDA_VER46N) || defined(CUDA_VER46P) || defined(CUDA_VER46Q)
  CSC(cudaMemcpy(d_field_sorted, field_sorted, Nf * sizeof(real), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_source_sorted, source_sorted, Ns * sizeof(real4), cudaMemcpyHostToDevice));
#elif defined(CUDA_VER46E)
  CSC(cudaMemcpy(d_field_sorted, field_sorted, Nf * sizeof(real3), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_phi_sorted, phi_sorted, Nf * sizeof(real), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_q_sorted, q_sorted, Ns * sizeof(real), cudaMemcpyHostToDevice));
#elif defined(CUDA_VER46C) || defined(CUDA_VER46D)
  CSC(cudaMemcpy(d_fieldlist, fieldlist, Nf * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_field, field, Nf * sizeof(real3), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_phi, phi, Nf * sizeof(real), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_q, q, Ns * sizeof(real), cudaMemcpyHostToDevice));
#elif defined(CUDA_VER46B)
  CSC(cudaMemcpy(d_fieldlist, fieldlist, Nf * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_sourcelist, sourcelist, Ns * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_sourcesta, sourcesta, ncell * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_sourceend, sourceend, ncell * sizeof(int), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_field4, field4, Nf * sizeof(real4), cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_source4, source4, Ns * sizeof(real4), cudaMemcpyHostToDevice));
#endif
  stopEventTimer(&time_direct_set);

  /* Obtain the number of cells in the maximum level; Note that all
     the leaves exist in the level of maxlev because non-adaptive tree
     is used */
  int nc = levend[maxlev] - levsta[maxlev] + 1;
  
  /* Obtain the number of cell-groups */
  int ngrp = (nc / DIRECT_GROUP_SIZE) + (nc % DIRECT_GROUP_SIZE == 0 ? 0 : 1);
  DBG("nc=%d ngp=%d\n", nc, ngrp);

  /* Statistics */
#if defined(MYDEBUG)
  double ave_num_field_per_cell = 0;
  double ave_num_source_per_cell = 0;
  int max_num_field_per_cell = 0;
  int max_num_source_per_cell = 0;
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) {
    int nf = (c->fieldend)[A] - (c->fieldsta)[A] + 1;
    int ns = (c->sourceend)[A] - (c->sourcesta)[A] + 1;
    ave_num_field_per_cell += nf;
    ave_num_source_per_cell += ns;
    max_num_field_per_cell = MAX(max_num_field_per_cell, nf);
    max_num_source_per_cell = MAX(max_num_source_per_cell, ns);
  }
  ave_num_field_per_cell /= nc;
  ave_num_source_per_cell /= nc;
  INFO("ave_num_field_per_cell = %6.1f\n", ave_num_field_per_cell);
  INFO("ave_num_source_per_cell = %6.1f\n", ave_num_source_per_cell);
  INFO("max_num_field_per_cell = %d\n", max_num_field_per_cell);
  INFO("max_num_source_per_cell = %d\n", max_num_source_per_cell);
#endif
  
  /* Loop for cell-groups */
  for (int grp = 0; grp < ngrp; grp ++) {
    
    /* Obtain the first and last cells of the current cell-group */
    int Asta = levsta[maxlev] + DIRECT_GROUP_SIZE * grp;
    int Aend = MIN(Asta + DIRECT_GROUP_SIZE - 1, levend[maxlev]);

    /* Number of cells in the current cell-group (<= DIRECT_GROUP_SIZE) */
    int ncg = Aend - Asta + 1;
    
    /* Perform direct computaions */
    startEventTimer(&time_direct_kernel);
#if defined(CUDA_VER46N) || defined(CUDA_VER46P) // field=source
    dim3 grid(ncg);
    dim3 block(DIRECT_NUM_THREADS_PER_BLOCK);
    direct<<<grid, block>>>(d_ineigh, d_neighbors, pitch_neighbors,
    			    d_fieldsta, d_fieldend,
			    d_field_sorted, d_source_sorted, Asta);
#elif defined(CUDA_VER46F) || defined(CUDA_VER46G) || defined(CUDA_VER46H) || defined(CUDA_VER46I) || defined(CUDA_VER46J) || defined(CUDA_VER46K) || defined(CUDA_VER46L) || defined(CUDA_VER46M) || defined(CUDA_VER46Q) // field=source & DIRECT_NUM_THREADS_PER_BLOCK>=SHARE_SIZE
    dim3 grid(ncg);
    dim3 block(DIRECT_NUM_THREADS_PER_BLOCK);
    direct<<<grid, block>>>(d_ineigh, d_neighbors, pitch_neighbors,
    			    d_fieldsta, d_fieldend,
			    d_field_sorted, d_source_sorted, Asta);
#elif defined(CUDA_VER46E) // field=source
    dim3 grid(ncg);
    dim3 block(DIRECT_NUM_THREADS_PER_BLOCK);
    direct<<<grid, block>>>(d_ineigh, d_neighbors, pitch_neighbors,
    			    d_fieldsta, d_fieldend,
			    d_field_sorted, d_phi_sorted, d_q_sorted, Asta);
#elif defined(CUDA_VER46C) || defined(CUDA_VER46D) // field=source
    dim3 grid(ncg);
    dim3 block(DIRECT_NUM_THREADS_PER_BLOCK);
    direct<<<grid, block>>>(d_ineigh, d_neighbors, pitch_neighbors,
			    d_fieldlist, d_fieldsta, d_fieldend,
			    d_field, d_phi, d_q, Asta);
#elif defined(CUDA_VER46B)
    dim3 grid(ncg);
    dim3 block(DIRECT_NUM_THREADS_PER_BLOCK); // 27 or more
    direct<<<grid, block>>>(d_ineigh, d_neighbors, pitch_neighbors,
			    d_fieldlist, d_fieldsta, d_fieldend,
			    d_sourcelist, d_sourcesta, d_sourceend,
			    d_field4, d_source4, Asta);
#endif
    stopEventTimer(&time_direct_kernel);

  }    

  /* Receive the field values */
  startEventTimer(&time_direct_get);
#if defined(CUDA_VER46F) || defined(CUDA_VER46G) || defined(CUDA_VER46H) || defined(CUDA_VER46I) || defined(CUDA_VER46J) || defined(CUDA_VER46K) || defined(CUDA_VER46L) || defined(CUDA_VER46M) || defined(CUDA_VER46N) || defined(CUDA_VER46P) || defined(CUDA_VER46Q)
  CSC(cudaMemcpy(field_sorted, d_field_sorted, Nf * sizeof(real), cudaMemcpyDeviceToHost));
#elif defined(CUDA_VER46E)
  CSC(cudaMemcpy(phi_sorted, d_phi_sorted, Nf * sizeof(real), cudaMemcpyDeviceToHost));
#elif defined(CUDA_VER46C) || defined(CUDA_VER46D)
  CSC(cudaMemcpy(phi, d_phi, Nf * sizeof(real), cudaMemcpyDeviceToHost)); // store directly
#elif defined(CUDA_VER46B)
  CSC(cudaMemcpy(field4, d_field4, Nf * sizeof(real4), cudaMemcpyDeviceToHost));
#endif
  stopEventTimer(&time_direct_get);

  /* Store the field values on CPU */
#if defined(CUDA_VER46F) || defined(CUDA_VER46G) || defined(CUDA_VER46H) || defined(CUDA_VER46I) || defined(CUDA_VER46J) || defined(CUDA_VER46K) || defined(CUDA_VER46L) || defined(CUDA_VER46M) || defined(CUDA_VER46N) || defined(CUDA_VER46P) || defined(CUDA_VER46Q)
#if defined(FAST_HOST_CODE)
  auxAnotherDownwardPass_read_sorted_field_values(levsta[maxlev], levend[maxlev],
  						  fieldlist, fieldsta, fieldend,
						  field_sorted, phi);
#else
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) {
    for (int i = fieldsta[A]; i <= fieldend[A]; i ++) {
      int findex = fieldlist[i];
      phi[findex] = field_sorted[i];
    }
  }
#endif
#elif defined(CUDA_VER46E)
#if defined(FAST_HOST_CODE)
  auxAnotherDownwardPass_read_sorted_field_values(levsta[maxlev], levend[maxlev],
						  fieldlist, fieldsta, fieldend,
						  phi_sorted, phi);
#else
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) {
    for (int i = fieldsta[A]; i <= fieldend[A]; i ++) {
      int findex = fieldlist[i];
      phi[findex] = phi_sorted[i];
    }
  }
#endif
#elif defined(CUDA_VER46B)
#if defined(FAST_HOST_CODE)
  read_phi_from_field4(Nf, (void *)field4, phi);
#else
  for (int i = 0; i < Nf; i++) {
    phi[i] = field4[i].w;
  }
#endif
#endif

  /* Free */
  CSC(cudaFree(d_ineigh));
  CSC(cudaFree(d_neighbors));
  CSC(cudaFree(d_fieldsta));
  CSC(cudaFree(d_fieldend));
#if defined(CUDA_VER46F) || defined(CUDA_VER46G) || defined(CUDA_VER46H) || defined(CUDA_VER46I) || defined(CUDA_VER46J) || defined(CUDA_VER46K) || defined(CUDA_VER46L) || defined(CUDA_VER46M) || defined(CUDA_VER46N) || defined(CUDA_VER46P) || defined(CUDA_VER46Q)
  CSC(cudaFree(d_field_sorted));
  CSC(cudaFree(d_source_sorted));
  free(field_sorted);
  free(source_sorted);
#elif defined(CUDA_VER46E)
  CSC(cudaFree(d_field_sorted));
  CSC(cudaFree(d_phi_sorted));
  CSC(cudaFree(d_q_sorted));
  free(field_sorted);
  free(phi_sorted);
  free(q_sorted);
#elif defined(CUDA_VER46C) || defined(CUDA_VER46D)
  CSC(cudaFree(d_fieldlist));
  CSC(cudaFree(d_field));
  CSC(cudaFree(d_phi));
  CSC(cudaFree(d_q));
#elif defined(CUDA_VER46B)
  CSC(cudaFree(d_fieldlist));
  CSC(cudaFree(d_sourcelist));
  CSC(cudaFree(d_sourcesta));
  CSC(cudaFree(d_sourceend));
  CSC(cudaFree(d_field4));
  CSC(cudaFree(d_source4));
  free(field4);
  free(source4);
#endif

  /* Finalise timers */
  printEventTimer(stderr, "time_direct_set", &time_direct_set);
  printEventTimer(stderr, "time_direct_kernel", &time_direct_kernel);
  printEventTimer(stderr, "time_direct_get", &time_direct_get);
  finalizeEventTimer(&time_direct_set);
  finalizeEventTimer(&time_direct_kernel);
  finalizeEventTimer(&time_direct_get);

#if defined(CHECK_PERFORMANCE)
  /* Calculate kernel's performance */
  double num_pairwise_interactions = 0;
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) {
    int nf = (c->fieldend)[A] - (c->fieldsta)[A] + 1;
    for (int i = 0; i < ineigh[A]; i ++) {
      int B = (c->neighbors)[c->pitch_neighbors * A + i];
      int ns = (c->sourceend)[B] - (c->sourcesta)[B] + 1;
      num_pairwise_interactions += nf * ns;
    }
  }
  double num_pairwise_interactions_per_sec = num_pairwise_interactions / getEventTimer(&time_direct_kernel);
  INFO("num_pairwise_interactions_per_sec = %f [G interaction/s]\n", num_pairwise_interactions_per_sec / giga);
#endif


#endif /* defined(ENABLE_NEARFIELD_BY_CPU) */


  /* Finalise the timer for direct computation */
  stopEventTimer(&time_direct_all);
  printEventTimer(stderr, "time_direct_all", &time_direct_all);
  finalizeEventTimer(&time_direct_all);

}
/**************************************************************************/
#else
/**************************************************************************/
#error USE VER46 OR LATER.
/**************************************************************************/
#endif
/**************************************************************************/

#endif /* ANOTHERDOWNWARDPASS_CU */
