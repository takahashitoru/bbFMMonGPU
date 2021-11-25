#include "bbfmm.h" // defines FIELDPOINTS_EQ_SOURCES

#if !defined(SLOW)

static int ibrnch(const real vx, const real vy, const real vz)
{
  int s = (vx < ZERO ? 0 : 4);
  s += (vy < ZERO ? 0 : 2);
  s += (vz < ZERO ? 0 : 1);
  return s;
}

#else

#ifdef SINGLE
typedef union {
  unsigned int i;
  float r;
} uir;
#define SIGN(x) (~(x) >> 31)
#else
typedef union {
  unsigned long int i;
  double r;
} uir;
#define SIGN(x) (~(x) >> 63)
#endif

static int ibrnch(real vx, real vy, real vz)
{
  uir x, y, z;

  x.r = vx;
  y.r = vy;
  z.r = vz;
  
  return (SIGN(x.i) << 2) + (SIGN(y.i) << 1) + SIGN(z.i);
}

#endif

#if(0) // original

#define FAST_MEMCPY
#if defined(FAST_MEMCPY)
#include "fast_memcpy.h"
#endif

static void sortList(int *list, const real3 *pos, int *sta, int *end, const int A, const real Ax, const real Ay, const real Az)
{
  /* Compute the first and last cells (leaves) at maximum level */
  const int staA = sta[A];
  const int endA = end[A];
  
  /* Number of particles in the relevant cell */
  const int nc = endA - staA + 1;
  
  /* Allocate counters and lists for children */
  int *cnum = (int *)calloc(8, sizeof(int)); // initialise
  int *clist = (int *)malloc(8 * nc * sizeof(int));
  
  /* Count the number of children according to their sibling-indexes */
  for (int i = staA; i <= endA; i ++) {
    const int particle = list[i];
    const int s = ibrnch(pos[particle].x - Ax, pos[particle].y - Ay, pos[particle].z - Az); // sibling-index
    clist[nc * s + cnum[s]] = particle;
    cnum[s] ++;
  }
  
  /* Obtain the starting and end address of child cells */
  int l = staA;
  int B = GET_CHILD_INDEX(A, 0); // 0-th child of A
  for (int s = 0; s < 8; s ++) {
    
#if defined(FAST_MEMCPY)
    sta[B] = l;
    fast_memcpy((void *)&(list[l]), (void *)&(clist[nc * s]), cnum[s] * sizeof(int));
    l += cnum[s];
    end[B] = l - 1;
#else
    sta[B] = l;
    for (int i = 0; i < cnum[s]; i ++) {
      list[l] = clist[nc * s + i];
      l ++;
    }
    end[B] = l - 1; // note that if cnum[c]=0, end[B]-sta[B]+1=0
#endif
    
    B ++; 
  }
  
  free(cnum);
  free(clist);
  
}


static void comp_child_center(real3 *center, const real Lquater,
			      const int A, const real Ax, const real Ay, const real Az)
{
  const real xp = Ax + Lquater;
  const real xm = Ax - Lquater;
  const real yp = Ay + Lquater;
  const real ym = Ay - Lquater;
  const real zp = Az + Lquater;
  const real zm = Az - Lquater;
  int B = GET_CHILD_INDEX(A, 0);
  center[B].x = xm; center[B].y = ym; center[B].z = zm; B ++; // 0th child
  center[B].x = xm; center[B].y = ym; center[B].z = zp; B ++; // 1st
  center[B].x = xm; center[B].y = yp; center[B].z = zm; B ++; // 2nd
  center[B].x = xm; center[B].y = yp; center[B].z = zp; B ++; // 3rd
  center[B].x = xp; center[B].y = ym; center[B].z = zm; B ++; // 4th
  center[B].x = xp; center[B].y = ym; center[B].z = zp; B ++; // 5th
  center[B].x = xp; center[B].y = yp; center[B].z = zm; B ++; // 6th
  center[B].x = xp; center[B].y = yp; center[B].z = zp;       // 7th
}

static void AnotherFMMDistribute(int maxlev, int *levsta, int *levend, real3 *center, real *celeng,
				 int Nf, int *fieldlist, int *fieldsta, int *fieldend, real3 *field)
{
  
  /* Initialise field and source lists */
  for (int i = 0; i < Nf; i ++) { // LOOP WAS VECTORIZED.
    fieldlist[i] = i;
  }
  
  /* Set up the root cell */
  fieldsta[0] = 0;
  fieldend[0] = Nf - 1;
  center[0].x = 0.0;
  center[0].y = 0.0;
  center[0].z = 0.0;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (int level = 0; level < maxlev; level ++) { // maxlev is excluded because of perfect tree
    
    const real Lquater = celeng[level] / 4;
    
    /* Loop over cells at this level */
#ifdef _OPENMP
#pragma omp for
#endif
    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
      
      /* Center of A */
      const real Ax = center[A].x;
      const real Ay = center[A].y;
      const real Az = center[A].z;
      
      /* Create children of A */
      comp_child_center(center, Lquater, A, Ax, Ay, Az);      
      
      /* Sort the particle list for A by child types */
      sortList(fieldlist, field, fieldsta, fieldend, A, Ax, Ay, Az);
      
    } // A
  } // level

}

#else // another algorithm

static void comp_child_center(real3 *center, const real Lquater, const int A)
{
  /* Obtain the center of A */
  const real Ax = center[A].x;
  const real Ay = center[A].y;
  const real Az = center[A].z;

  /* Compute the centers of children of A */
  const real xp = Ax + Lquater;
  const real xm = Ax - Lquater;
  const real yp = Ay + Lquater;
  const real ym = Ay - Lquater;
  const real zp = Az + Lquater;
  const real zm = Az - Lquater;
  int B = GET_CHILD_INDEX(A, 0);
  center[B].x = xm; center[B].y = ym; center[B].z = zm; B ++; // 0th child
  center[B].x = xm; center[B].y = ym; center[B].z = zp; B ++; // 1st
  center[B].x = xm; center[B].y = yp; center[B].z = zm; B ++; // 2nd
  center[B].x = xm; center[B].y = yp; center[B].z = zp; B ++; // 3rd
  center[B].x = xp; center[B].y = ym; center[B].z = zm; B ++; // 4th
  center[B].x = xp; center[B].y = ym; center[B].z = zp; B ++; // 5th
  center[B].x = xp; center[B].y = yp; center[B].z = zm; B ++; // 6th
  center[B].x = xp; center[B].y = yp; center[B].z = zp;       // 7th
}


static void comp_center(real3 *center, const real Lhalf, const int A)
{
  /* Obtain the parent cell P */
  const int P = GET_PARENT_INDEX(A);
  
  /* Obtain the sibling-index of A */
  int s = A - GET_CHILD_INDEX(P, 0);
  
  /* Compute the center of A */
#if(0)
  if (s < 4) {
    center[A].x = center[P].x - Lhalf;
    if (s < 2) {
      center[A].y = center[P].y - Lhalf;
      if (s) { // s=1
	center[A].z = center[P].z + Lhalf;
      } else { // s=0
	center[A].z = center[P].z - Lhalf;
      }
    } else {
      center[A].y = center[P].y + Lhalf;
      if (s == 2) { // s=2
	center[A].z = center[P].z - Lhalf;
      } else { // s=3
	center[A].z = center[P].z + Lhalf;
      }
    }
  } else {
    center[A].x = center[P].x + Lhalf;
    s = s % 4;
    if (s < 2) {
      center[A].y = center[P].y - Lhalf;
      if (s) { // s=5
	center[A].z = center[P].z + Lhalf;
      } else { // s=4
	center[A].z = center[P].z - Lhalf;
      }
    } else {
      center[A].y = center[P].y + Lhalf;
      if (s == 2) { // s=6
	center[A].z = center[P].z - Lhalf;
      } else { // s=7
	center[A].z = center[P].z + Lhalf;
      }
    }
  }    

#else
  if (s < 4) {
    center[A].x = center[P].x - Lhalf; // s=0,1,2,3
  } else {
    center[A].x = center[P].x + Lhalf; // s=4,5,6,7
  }
  if (s % 4 < 2) {
    center[A].y = center[P].y - Lhalf; // s=0,1,4,5
  } else {
    center[A].y = center[P].y + Lhalf; // s=2,3,6,7
  }
  if (s % 2 == 0) {
    center[A].z = center[P].z - Lhalf; // s=0,2,4,6
  } else {
    center[A].z = center[P].z + Lhalf; // s=1,3,5,7
  }
#endif
}


static int compute_xyz_cell_index(const real L0half, const real iLm,
				  const int ncpd, const real x, const real y, const real z)
{
  /*
    Note: It is certain that ix,iy,iz>=0 even for round-off error of LHS,
    but x,y,z=L0half can happen (then ix,iy,iz can become ncpd).
  */
#if(1)
  //  const int ncpd1 = ncpd - 1;
  int ix = MIN((int)((x + L0half) * iLm), ncpd - 1);
  int iy = MIN((int)((y + L0half) * iLm), ncpd - 1);
  int iz = MIN((int)((z + L0half) * iLm), ncpd - 1);
#else
  int ix = (x + L0half) * iLm;
  ix = MIN(ix, ncpd - 1);
  int iy = (y + L0half) * iLm;
  iy = MIN(iy, ncpd - 1);
  int iz = (z + L0half) * iLm;
  iz = MIN(iz, ncpd - 1);
#endif

  //////////////////////////////////////////////////////
#if(0)
  if (ix < 0 || ix >= ncpd) {
    INFO("Invalid ix=%d: x=%e y=%e z=%e\n", ix, x, y, z);
    exit(EXIT_FAILURE);
  }
  if (iy < 0 || iy >= ncpd) {
    INFO("Invalid iy=%d: x=%e y=%e z=%e\n", iy, x, y, z);
    exit(EXIT_FAILURE);
  }
  if (iz < 0 || iz >= ncpd) {
    INFO("Invalid iz=%d: x=%e y=%e z=%e\n", iz, x, y, z);
    exit(EXIT_FAILURE);
  }
#endif
  //////////////////////////////////////////////////////


  return (iz * ncpd + iy) * ncpd + ix;
}

static int compute_xyz_cell_index_safe(const real L0half, const real iLm,
				       const int ncpd, const real x, const real y, const real z)
{
  /* This function supposes that x,y,z<L0half so that
     ix,iy,iz<=ncpd-1 */
  int ix = (int)((x + L0half) * iLm);
  int iy = (int)((y + L0half) * iLm);
  int iz = (int)((z + L0half) * iLm);
  return (iz * ncpd + iy) * ncpd + ix;
}


//120208#if(1) //ver2

static void AnotherFMMDistribute(const int maxlev, const int *levsta, const int *levend,
				 real3 *center, const real *celeng,
				 const int N, int *list, int *sta, int *end, const real3 *pos)
{
  /* Note that only {sta,end}[levsta[maxlev]:levend[maxlev]] is
     computed here */
  
  /* Set up the root cell */
  center[0].x = ZERO;
  center[0].y = ZERO;
  center[0].z = ZERO;
  
#if(0)
  timerType *timer_setup_tree_center;
  allocTimer(&timer_setup_tree_center);
  initTimer(timer_setup_tree_center);
  startTimer(timer_setup_tree_center);
#endif
  /* Compute the centers of all the cells */
#ifdef _OPENMP
#pragma omp parallel
#endif
  for (int level = 1; level <= maxlev; level ++) {
    const real Lhalf = celeng[level] / 2;
#ifdef _OPENMP
#pragma omp for
#endif
    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
      comp_center(center, Lhalf, A);
    }
  }
#if(0)
  stopTimer(timer_setup_tree_center);
  printTimer(stderr, "setup_tree_center", timer_setup_tree_center);
  freeTimer(&timer_setup_tree_center);
#endif

  /* Compute the first and last cells (leaves) at maximum level */
  const int Asta = levsta[maxlev];
  const int Aend = levend[maxlev];
  
  /* Compute the mapping from the normal cell index to the xyz-base
     cell index */
  const int ncpd = POW2(maxlev); // number of cells (leaves) per dimension, that is, 2^maxlev
  int *map = (int *)malloc(ncpd * ncpd * ncpd * sizeof(int)); // 8^maxlev
  const real L0half = celeng[0] / 2;
  const real iLm = 1 / celeng[maxlev];
  
  /* Compute the number of cells (leaves) at maximum level */
  const int nc = Aend - Asta + 1; // this is 8^maxlev
  
  /* Obtain the maximum number of threads */
  const int nthreads = omp_get_max_threads();

  /* Allocate the counters of numbers of particles in a leaf */
  int *stas = (int *)calloc(nthreads * nc, sizeof(int)); // stass[nthreads][nc] initialize

  /* Allocate the array to map the patricle index to a certain index */
  int *loc = (int *)malloc(N * sizeof(int));

#if(0)
  timerType *timer_setup_tree_count;
  allocTimer(&timer_setup_tree_count);
  initTimer(timer_setup_tree_count);
  startTimer(timer_setup_tree_count);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
#pragma omp for
#endif
    /* Loop over leaves */
    for (int A = Asta; A <= Aend; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
      const int m = compute_xyz_cell_index_safe(L0half, iLm, ncpd, center[A].x, center[A].y, center[A].z);
      map[m] = A;
    }
  
    /* Loop over particles */
#ifdef _OPENMP
#pragma omp for
#endif
    for (int i = 0; i < N; i ++) {
      
      /* Compute the leaf index A that the particle i belongs to */
      const int m = compute_xyz_cell_index(L0half, iLm, ncpd, pos[i].x, pos[i].y, pos[i].z);
      const int A = map[m];
      
      /* Obtain the thread ID */
      const int tid = omp_get_thread_num();

      /* Increase the counter */
      const int k = nc * tid + (A - Asta);
      stas[k] ++;

      /* Store the index */
      loc[i] = k;
    }
  }

#if(0)
  stopTimer(timer_setup_tree_count);
  printTimer(stderr, "setup_tree_count", timer_setup_tree_count);
  freeTimer(&timer_setup_tree_count);
#endif

  free(map);

#if(0)
  timerType *timer_setup_tree_address;
  allocTimer(&timer_setup_tree_address);
  initTimer(timer_setup_tree_address);
  startTimer(timer_setup_tree_address);
#endif

  /* Compute the starting and end addresses of the particle-list for
     each leaf */
  int l = 0;
  for (int A = Asta; A <= Aend; A ++) {
    sta[A] = l;
    int k = A - Asta; // k=nc*tid+(A-Asta)
    for (int tid = 0; tid < nthreads; tid ++) {

      const int ltmp = l; // stas[k]=l is not allowed because stas is identical to pnums
      l += stas[k];
      stas[k] = ltmp;

      k += nc;

    }
    end[A] = l - 1;
  }

#if(0)
  stopTimer(timer_setup_tree_address);
  printTimer(stderr, "setup_tree_address", timer_setup_tree_address);
  freeTimer(&timer_setup_tree_address);
#endif

#if(0)
  timerType *timer_setup_tree_list;
  allocTimer(&timer_setup_tree_list);
  initTimer(timer_setup_tree_list);
  startTimer(timer_setup_tree_list);
#endif

#ifdef _OPENMP
#pragma omp for
#endif
  for (int i = 0; i < N; i ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
    
    /* Obtain the index for pnums and stas */
    const int k = loc[i]; // nc*tid[i]+(belong[i]-Asta)

    /* Append the particle to the particle-list and icnrease the address */
    list[stas[k] ++] = i;

  }

#if(0)
  stopTimer(timer_setup_tree_list);
  printTimer(stderr, "setup_tree_list", timer_setup_tree_list);
  freeTimer(&timer_setup_tree_list);
#endif

  free(stas);
  free(loc);

}

//120208#else //ver1
//120208
//120208static void AnotherFMMDistribute(const int maxlev, const int *levsta, const int *levend,
//120208				 real3 *center, const real *celeng,
//120208				 const int N, int *list, int *sta, int *end, const real3 *pos)
//120208{
//120208  /* Note that only {sta,end}[levsta[maxlev]:levend[maxlev]] is
//120208     computed here */
//120208  
//120208  /* Set up the root cell */
//120208  center[0].x = ZERO;
//120208  center[0].y = ZERO;
//120208  center[0].z = ZERO;
//120208  
//120208#ifndef DISABLE_TIMING
//120208  timerType *timer_setup_tree_center;
//120208  allocTimer(&timer_setup_tree_center);
//120208  initTimer(timer_setup_tree_center);
//120208  startTimer(timer_setup_tree_center);
//120208#endif
//120208  /* Compute the centers of all the cells */
//120208#ifdef _OPENMP
//120208#pragma omp parallel
//120208#endif
//120208  for (int level = 1; level <= maxlev; level ++) {
//120208    const real Lhalf = celeng[level] / 2;
//120208#ifdef _OPENMP
//120208#pragma omp for
//120208#endif
//120208    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
//120208      comp_center(center, Lhalf, A);
//120208    }
//120208  }
//120208#ifndef DISABLE_TIMING
//120208  stopTimer(timer_setup_tree_center);
//120208  printTimer(stderr, "setup_tree_center", timer_setup_tree_center);
//120208  freeTimer(&timer_setup_tree_center);
//120208#endif
//120208
//120208  /* Compute the first and last cells (leaves) at maximum level */
//120208  const int Asta = levsta[maxlev];
//120208  const int Aend = levend[maxlev];
//120208  
//120208  /* Compute the mapping from the normal cell index to the xyz-base
//120208     cell index */
//120208  const int ncpd = POW2(maxlev); // number of cells (leaves) per dimension, that is, 2^maxlev
//120208  int *map = (int *)malloc(ncpd * ncpd * ncpd * sizeof(int)); // 8^maxlev
//120208  const real L0half = celeng[0] / 2;
//120208  const real iLm = 1 / celeng[maxlev];
//120208  
//120208  /* Compute the number of cells (leaves) at maximum level */
//120208  const int nc = Aend - Asta + 1; // this is 8^maxlev
//120208  
//120208  /* Obtain the maximum number of threads */
//120208  const int nthreads = omp_get_max_threads();
//120208
//120208  /* Allocate the counters of numbers of particles in a leaf */
//120208  int *pnums = (int *)calloc(nthreads * nc, sizeof(int)); // pnums[nthreads][nc] initialize
//120208
//120208  int *loc = (int *)malloc(N * sizeof(int));
//120208
//120208#ifndef DISABLE_TIMING
//120208  timerType *timer_setup_tree_count;
//120208  allocTimer(&timer_setup_tree_count);
//120208  initTimer(timer_setup_tree_count);
//120208  startTimer(timer_setup_tree_count);
//120208#endif
//120208#ifdef _OPENMP
//120208#pragma omp parallel
//120208#endif
//120208  {
//120208#ifdef _OPENMP
//120208#pragma omp for
//120208#endif
//120208    /* Loop over leaves */
//120208    for (int A = Asta; A <= Aend; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
//120208      const int m = compute_xyz_cell_index_safe(L0half, iLm, ncpd, center[A].x, center[A].y, center[A].z);
//120208      map[m] = A;
//120208    }
//120208  
//120208    /* Loop over particles */
//120208#ifdef _OPENMP
//120208#pragma omp for
//120208#endif
//120208    for (int i = 0; i < N; i ++) {
//120208      
//120208      /* Compute the leaf index A that the particle i belongs to */
//120208      const int m = compute_xyz_cell_index(L0half, iLm, ncpd, pos[i].x, pos[i].y, pos[i].z);
//120208      const int A = map[m];
//120208      
//120208      /* Obtain the thread ID */
//120208      const int tid = omp_get_thread_num();
//120208
//120208      const int k = nc * tid + (A - Asta);
//120208      pnums[k] ++;
//120208
//120208      loc[i] = k;
//120208    }
//120208  }
//120208#ifndef DISABLE_TIMING
//120208  stopTimer(timer_setup_tree_count);
//120208  printTimer(stderr, "setup_tree_count", timer_setup_tree_count);
//120208  freeTimer(&timer_setup_tree_count);
//120208#endif
//120208
//120208  free(map);
//120208
//120208  /* Allocate the counters of numbers of particles in a leaf */
//120208  int *stas = pnums; // just a link
//120208
//120208#ifndef DISABLE_TIMING
//120208  timerType *timer_setup_tree_address;
//120208  allocTimer(&timer_setup_tree_address);
//120208  initTimer(timer_setup_tree_address);
//120208  startTimer(timer_setup_tree_address);
//120208#endif
//120208  /* Compute the starting and end addresses of the particle-list for
//120208     each leaf */
//120208  int l = 0;
//120208  for (int A = Asta; A <= Aend; A ++) {
//120208    sta[A] = l;
//120208    int k = A - Asta; // k=nc*tid+(A-Asta)
//120208    for (int tid = 0; tid < nthreads; tid ++) {
//120208
//120208      const int ltmp = l; // stas[k]=l is not allowed because stas is identical to pnums
//120208      l += pnums[k];
//120208      stas[k] = ltmp;
//120208
//120208      k += nc;
//120208
//120208    }
//120208    end[A] = l - 1;
//120208  }
//120208
//120208#ifndef DISABLE_TIMING
//120208  stopTimer(timer_setup_tree_address);
//120208  printTimer(stderr, "setup_tree_address", timer_setup_tree_address);
//120208  freeTimer(&timer_setup_tree_address);
//120208#endif
//120208
//120208#ifndef DISABLE_TIMING
//120208  timerType *timer_setup_tree_list;
//120208  allocTimer(&timer_setup_tree_list);
//120208  initTimer(timer_setup_tree_list);
//120208  startTimer(timer_setup_tree_list);
//120208#endif
//120208  /* Compute the particle-list */
//120208#ifdef _OPENMP
//120208#pragma omp for
//120208#endif
//120208  for (int i = 0; i < N; i ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
//120208    
//120208    /* Obtain the index for pnums and stas */
//120208    const int k = loc[i]; // nc*tid[i]+(belong[i]-Asta)
//120208
//120208    /* Append the particle to the particle-list and icnrease the address */
//120208    list[stas[k] ++] = i;
//120208
//120208  }
//120208#ifndef DISABLE_TIMING
//120208  stopTimer(timer_setup_tree_list);
//120208  printTimer(stderr, "setup_tree_list", timer_setup_tree_list);
//120208  freeTimer(&timer_setup_tree_list);
//120208#endif
//120208
//120208  free(pnums);
//120208  free(loc);
//120208}
//120208
//120208#endif
//120208
//120208#if(0) //ver0
//120208
//120208static void AnotherFMMDistribute(const int maxlev, const int *levsta, const int *levend,
//120208				 real3 *center, const real *celeng,
//120208				 const int N, int *list, int *sta, int *end, const real3 *pos)
//120208{
//120208  /* Note that only {sta,end}[levsta[maxlev]:levend[maxlev]] is
//120208     computed here */
//120208  
//120208  /* Set up the root cell */
//120208  center[0].x = ZERO;
//120208  center[0].y = ZERO;
//120208  center[0].z = ZERO;
//120208  
//120208  /* Compute the centers of all the cells */
//120208#if(1)
//120208
//120208#ifdef _OPENMP
//120208#pragma omp parallel
//120208#endif
//120208  for (int level = 1; level <= maxlev; level ++) {
//120208    const real Lhalf = celeng[level] / 2;
//120208#ifdef _OPENMP
//120208#pragma omp for
//120208#endif
//120208    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
//120208      comp_center(center, Lhalf, A);
//120208    }
//120208  }
//120208
//120208#else
//120208
//120208#ifdef _OPENMP
//120208#pragma omp parallel
//120208#endif
//120208  for (int level = 0; level < maxlev; level ++) { // maxlev is excluded because of perfect tree
//120208    const real Lquater = celeng[level] / 4;
//120208#ifdef _OPENMP
//120208#pragma omp for
//120208#endif
//120208    for (int A = levsta[level]; A <= levend[level]; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
//120208      comp_child_center(center, Lquater, A);
//120208    }
//120208  }
//120208
//120208#endif
//120208    
//120208  /* Compute the first and last cells (leaves) at maximum level */
//120208  const int Asta = levsta[maxlev];
//120208  const int Aend = levend[maxlev];
//120208  
//120208  /* Compute the mapping from the normal cell index to the xyz-base
//120208     cell index */
//120208  const int ncpd = POW2(maxlev); // number of cells (leaves) per dimension, that is, 2^maxlev
//120208  int *map = (int *)malloc(ncpd * ncpd * ncpd * sizeof(int)); // 8^maxlev
//120208  const real L0half = celeng[0] / 2;
//120208  const real iLm = 1 / celeng[maxlev];
//120208  
//120208  /* Compute the number of cells (leaves) at maximum level */
//120208  const int nc = Aend - Asta + 1;
//120208  
//120208  /* Allocate the counters of numbers of particles in a leaf */
//120208  int *pnum = (int *)calloc(nc, sizeof(int)); // initialize
//120208
//120208
//120208#define BELONG
//120208#if defined(BELONG)
//120208  int *belong = (int *)malloc(N * sizeof(int));
//120208#endif
//120208
//120208#ifdef _OPENMP
//120208#pragma omp parallel
//120208#endif
//120208  {
//120208#ifdef _OPENMP
//120208#pragma omp for
//120208#endif
//120208    /* Loop over leaves */
//120208    for (int A = Asta; A <= Aend; A ++) { // OpenMP DEFINED LOOP WAS PARALLELIZED.
//120208      const int m = compute_xyz_cell_index(L0half, iLm, ncpd, center[A].x, center[A].y, center[A].z);
//120208      map[m] = A;
//120208    }
//120208  
//120208    /* Loop over particles */
//120208#ifdef _OPENMP
//120208#pragma omp for
//120208#endif
//120208    for (int i = 0; i < N; i ++) {
//120208    
//120208      /* Compute the leaf index that this particle belongs to */
//120208      const int k = compute_xyz_cell_index(L0half, iLm, ncpd, pos[i].x, pos[i].y, pos[i].z);
//120208      const int A = map[k];
//120208
//120208#if defined(BELONG)
//120208      belong[i] = A;
//120208#endif
//120208    
//120208      /* Increase the counter of A */
//120208#ifdef _OPENMP
//120208#pragma omp atomic
//120208#endif
//120208      pnum[A - Asta] ++; // OpenMP multithreaded code generation for ATOMIC was successful.
//120208      
//120208    }
//120208  }
//120208
//120208
//120208  /* Compute the starting and end addresses of the particle-list for
//120208     each leaf */
//120208#if(1)
//120208  int *staA = &(sta[Asta]);
//120208  int *endA = &(end[Asta]);
//120208  int l = 0;
//120208  for (int i = 0; i < nc; i ++) {
//120208    staA[i] = l;
//120208    endA[i] = l + pnum[i] - 1;
//120208    l += pnum[i];
//120208  }
//120208#else
//120208  int l = 0;
//120208  for (int A = Asta; A <= Aend; A ++) {
//120208    sta[A] = l;
//120208    end[A] = sta[A] + pnum[A - Asta] - 1;
//120208    l += pnum[A - Asta];
//120208  }
//120208#endif
//120208  
//120208  /* Reset the couters */
//120208  memset(pnum, 0, nc * sizeof(int));
//120208  
//120208  /* Compute the particle-list */
//120208  //#ifdef _OPENMP
//120208  //#pragma omp for
//120208  //#endif
//120208  for (int i = 0; i < N; i ++) {
//120208    
//120208    /* Compute the leaf index that the i-th particle belongs to */
//120208#if defined(BELONG)
//120208    const int A = belong[i];
//120208#else
//120208    const int k = compute_xyz_cell_index(L0half, iLm, ncpd, pos[i].x, pos[i].y, pos[i].z);
//120208    const int A = map[k];
//120208#endif
//120208    
//120208    //#ifdef _OPENMP
//120208    //#pragma omp critical
//120208    //#endif
//120208    { // OpenMP multithreaded code generation for CRITICAL was successful.
//120208      /* Append the particle to the particle-list */
//120208      list[sta[A] + pnum[A - Asta]] = i;
//120208      
//120208      /* Increase the counter of A */
//120208      pnum[A - Asta] ++;
//120208    }
//120208    
//120208  }
//120208
//120208#if defined(BELONG)
//120208  free(belong);
//120208#endif
//120208  
//120208  free(pnum);
//120208  free(map);
//120208  
//120208}
//120208
//120208#endif

#endif


#ifdef __INTEL_COMPILER
#pragma intel optimization_level 0
#endif
void anotherFMMDistribute(anotherTree **atree, real3 *field, real3 *source)
{
  int ncell = (*atree)->ncell;
  int maxlev = (*atree)->maxlev;
  int *levsta = (*atree)->levsta;
  int *levend = (*atree)->levend;
  real *celeng = (*atree)->celeng;
  cell *c = (*atree)->c;

  int Nf = (*atree)->Nf;
  (*atree)->fieldlist = (int *)malloc(Nf * sizeof(int));
  c->fieldsta = (int *)malloc(ncell * sizeof(int));
  c->fieldend = (int *)malloc(ncell * sizeof(int));
  AnotherFMMDistribute(maxlev, levsta, levend, c->center, celeng,
		       Nf, (*atree)->fieldlist, c->fieldsta, c->fieldend, field);
#if defined(FIELDPOINTS_EQ_SOURCES)
  (*atree)->sourcelist = (*atree)->fieldlist;
  c->sourcesta = c->fieldsta; // link only
  c->sourceend = c->fieldend; // link only
#else
  int Ns = (*atree)->Ns;
  (*atree)->sourcelist = (int *)malloc(Ns * sizeof(int));
  c->sourcesta = (int *)malloc(ncell * sizeof(int));
  c->sourceend = (int *)malloc(ncell * sizeof(int));
  AnotherFMMDistribute(maxlev, levsta, levend, c->center, celeng, 
		       Ns, (*atree)->sourcelist, c->sourcesta, c->sourceend, source); // center is recomputed
#endif
}



void anotherFMMDistribute_check(anotherTree *atree)
{
  int maxlev = atree->maxlev;
  int *levsta = atree->levsta;
  int *levend = atree->levend;
  cell *c = atree->c;

  int Nf = atree->Nf;
  int *fieldlist = atree->fieldlist;
  int *fieldsta = c->fieldsta;
  int *fieldend = c->fieldend;

  for (int level = 0; level <= maxlev; level ++) {
    int sumf = 0; // number of field points in the relevant level
    for (int A = levsta[level]; A <= levend[level]; A ++) {
      int nf = fieldend[A] - fieldsta[A] + 1; // number of field points in the relevant cell
      INFO("field: level=%d A=%d (%d):", level, A, nf);
      for (int i = fieldsta[A]; i <= fieldend[A]; i ++) {
	fprintf(stderr, " %d", fieldlist[i]);
      }
      fprintf(stderr, "\n");
      sumf += nf;
    }
    ASSERT(sumf == Nf);
  }
}


void anotherFMMDistribute_check2(anotherTree *atree)
{
  const int maxlev = atree->maxlev;
  const int *levsta = atree->levsta;
  const int *levend = atree->levend;
  const cell *c = atree->c;

  const int Nf = atree->Nf;
  const int *fieldlist = atree->fieldlist;
  const int *fieldsta = c->fieldsta;
  const int *fieldend = c->fieldend;

  /* Statistics */

  const int nc = levend[maxlev] - levsta[maxlev] + 1; // number of leaves
  double ave_num_particles_per_leaf = 0;
  int max_num_particles_per_leaf = 0;
  for (int A = levsta[maxlev]; A <= levend[maxlev]; A ++) {
    int num = fieldend[A] - fieldsta[A] + 1;
    ave_num_particles_per_leaf += num;
    max_num_particles_per_leaf = MAX(max_num_particles_per_leaf, num);
  }
  ave_num_particles_per_leaf /= nc;
  INFO("ave_num_particles_per_leaf = %6.1f\n", ave_num_particles_per_leaf);
  INFO("max_num_particles_per_leaf = %d\n", max_num_particles_per_leaf);
}
