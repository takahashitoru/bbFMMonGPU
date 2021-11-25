/*
 * File: bbfmm.c
 * Description: File contains functions used in the implementation of the
 * black-box fast multipole method.
 * ----------------------------------------------------------------------
 *
 * Black-Box Fast Multipole Method (BBFMM)
 * William Fong
 * Stanford University
 *
 */

#include "bbfmm.h"

void bbfmm(real3 *field, real3 *source, real *q, int Nf, int Ns,
	   real L, int n, real *phi, int l)
{
#ifndef DISABLE_TIMING
  startTimer(timer_setup);
#endif
  
  int dof = 1;    // Number of degrees of freedom (all scalars)
  real homogen;   // Order of kernel homogeneity
  int cutoff;     // Number of singular values to keep
  char Kmat[50];  // File name for kernel interaction matrix K
  char Umat[50];  // File name for matrix of singular vectors U
  
  int n2 = n * n;
  int n3 = n * n * n;
  int dofn3 = dof * n3;
  
  real *Kweights = (real *)malloc(n3 * sizeof(real));
  real *Cweights = (real *)malloc(2 * n2 * sizeof(real));
  real *Tkz      = (real *)malloc(n2 * sizeof(real));
  int Ktable[343];
 
  anotherTree *atree;

  /* Set up for FMM */
  FMMSetup(Tkz, Ktable, Kweights, Cweights, L, &homogen, &cutoff, n, dof, Ns, &l, Kmat, Umat);
  
  /* Initialize arrays */
  real *K = (real *)malloc(316 * cutoff * cutoff * sizeof(real));
  real *U = (real *)malloc(cutoff * dofn3 * sizeof(real));

  /* Read kernel interaction matrix K and matrix of singular vectors U */
#ifndef DISABLE_TIMING
  timerType *timer_setup_read;
  allocTimer(&timer_setup_read);
  initTimer(timer_setup_read);
  startTimer(timer_setup_read);
#endif
  FMMReadMatrices(K, U, cutoff, n, dof, Kmat, Umat);
#ifndef DISABLE_TIMING
  stopTimer(timer_setup_read);
  printTimer(stderr, "setup_read", timer_setup_read);
  freeTimer(&timer_setup_read);
#endif
    
  /* Create an octtree of another type */
  anotherBuildFMMHierachy(L, n, dof, l, cutoff, &atree, Nf, Ns, field, source);
#if(0)
  checkTree(atree);
#endif
  
#ifndef DISABLE_TIMING
  stopTimer(timer_setup);
  startTimer(timer_comp);
#endif

  /* Compute the field using BBFMM */
  anotherFMMCompute(&atree, field, source, q, K, U, Tkz, Ktable, Kweights, Cweights, homogen, cutoff, n, dof, phi);

  /* Clean an octtree of another type */
  anotherFMMCleanup(&atree);

  free(Kweights);
  free(Cweights);
  free(Tkz);

  free(K);
  free(U);

#ifndef DISABLE_TIMING
  stopTimer(timer_comp);
#endif
}

/*
 * Function: FMMSetup
 * -----------------------------------------------------------------
 * Prepare for the FMM calculation by setting the parameters, computing
 * the weight matrices, pre-computing the SVD (if necessary), reading 
 * in the necessary matrices, and building the FMM hierarchy.
 */
void FMMSetup(real *Tkz, int *Ktable, real *Kweights,
	      real *Cweights, real L, real *homogen, int *cutoff, 
	      int n, int dof, int Ns, int *l, char *Kmat, char *Umat)
{
#ifdef ENABLE_ASCII_IO
  char ext[] = "out";
#else
  char ext[] = "bin";
#endif

  /*********************************
   *    LAPLACIAN KERNEL (1/r)     *
   *********************************/
#ifdef LAPLACIAN
  *homogen = 1;
#ifdef SINGLE
  sprintf(Kmat, "slaplacianK%d.%s", n, ext);
  sprintf(Umat, "slaplacianU%d.%s", n, ext);
#else
  sprintf(Kmat, "laplacianK%d.%s", n, ext);
  sprintf(Umat, "laplacianU%d.%s", n, ext);
#endif

  /****************************************************
   * LAPLACIAN FORCE (r.x/r^3) - only the x-component *
   ****************************************************/
#elif LAPLACIANFORCE
  *homogen = 2;
#ifdef SINGLE
  sprintf(Kmat, "slaplacianforceK%d.%s", n, ext);
  sprintf(Umat, "slaplacianforceU%d.%s", n, ext);
#else
  sprintf(Kmat, "laplacianforceK%d.%s", n, ext);
  sprintf(Umat, "laplacianforceU%d.%s", n, ext);
#endif

  /*********************
   *    1/r^4 KERNEL   *
   *********************/
#elif ONEOVERR4
  *homogen = 4;
#ifdef SINGLE
  sprintf(Kmat, "soneoverr4K%d.%s", n, ext);
  sprintf(Umat, "soneoverr4U%d.%s", n, ext);
#else
  sprintf(Kmat, "oneoverr4K%d.%s", n, ext);
  sprintf(Umat, "oneoverr4U%d.%s", n, ext);
#endif
#endif

  /* Compute the number of levels in the FMM hierarchy */
  //  *l = (int)log(Ns)/log(8.0);
  //  if (*l > 0)
  //    *l = *l - 1;

  *l += (int)log(Ns)/log(8.0); // this is different from (int)(log(N)/log(8.0)); *l has certain number
  if (*l > 0) {
    *l = *l - 1;
  }
  
  /* Reset the maximum level so that it is equal or greter than the
     minimum level */
  if (*l < MINLEV) {
    *l = MINLEV;
  }

  INFO("l = %d\n", *l);

  /* Compute the Chebyshev weights and sets up the lookup table */
#ifndef DISABLE_TIMING
  timerType *timer_setup_weight;
  allocTimer(&timer_setup_weight);
  initTimer(timer_setup_weight);
  startTimer(timer_setup_weight);
#endif
  ComputeWeights(Tkz, Ktable, Kweights, Cweights,n);
#ifndef DISABLE_TIMING
  stopTimer(timer_setup_weight);
  printTimer(stderr, "setup_weight", timer_setup_weight);
  freeTimer(&timer_setup_weight);
#endif

  /* Precompute the SVD of the kernel interaction matrix (if necessary) */
#ifndef DISABLE_TIMING
  timerType *timer_setup_svd;
  allocTimer(&timer_setup_svd);
  initTimer(timer_setup_svd);
  startTimer(timer_setup_svd);
#endif
  FILE *fK, *fU;
  if ((fK = fopen(Kmat, "r")) == NULL || (fU = fopen(Umat, "r")) == NULL) {
    ComputeKernelSVD(Kweights,n,dof,Kmat,Umat);
    MESG("Executed ComputeKernelSVD.\n");
  } else {
    fclose(fK);
    fclose(fU);
  }
#ifndef DISABLE_TIMING
  stopTimer(timer_setup_svd);
  printTimer(stderr, "setup_svd", timer_setup_svd);
  freeTimer(&timer_setup_svd);
#endif

  FILE *f = fopen(Umat, "r");
#ifdef ENABLE_ASCII_IO
  fscanf(f, "%d", cutoff);
#else
  fread(cutoff, sizeof(int), 1, f);
#endif
  fclose(f);

  INFO("cutoff = %d\n", *cutoff);
}

/*
 * Function: FMMReadMatrices
 * ------------------------------------------------------------------
 * Read in the kernel interaction matrix M and the matrix of singular
 * vectors U.
 */
void FMMReadMatrices(real *K, real *U, int cutoff, int n, int dof, char *Kmat, char *Umat)
{
  int Ksize = 316 * cutoff * cutoff;
  int Usize = cutoff * dof * n * n * n;
 
  /* Read in kernel interaction matrix K */
  FILE *fK = fopen(Kmat, "r");
#ifdef ENABLE_ASCII_IO
  for (int i = 0; i < Ksize; i ++) {
#ifdef SINGLE
    fscanf(fK, "%f", &K[i]);
#else
    fscanf(fK, "%lf", &K[i]);
#endif
  }
#else
  fread(K, sizeof(real), Ksize, fK);
#endif
  fclose(fK);

  /* Read in matrix of singular vectors U */
  FILE *fU = fopen(Umat, "r");
#ifdef ENABLE_ASCII_IO
  int idummy;
  fscanf(fU, "%d", &idummy); // Read cutoff and discard
  for (int i = 0; i < Usize; i ++) {
#ifdef SINGLE
    fscanf(fU, "%f", &U[i]);
#else
    fscanf(fU, "%lf", &U[i]);
#endif
  }
#else
  int idummy;
  fread(&idummy, sizeof(int), 1, fU); // Read cutoff and discard it immediately
  fread(U, sizeof(real), Usize, fU);
#endif
  fclose(fU);
}

/*
 * Function: ReadParam
 * ------------------------------------------------------------------
 * Read in the user-specified options from the options file.
 * 
 */
int SetParam(char *options, fmmparam *param) {
  char tag[10], line[80];
  int i, tag_id, ntags=13;
  char alltags[][10]={"Ns","Nf","dof","L","n","levels","PBClevels",
		      "PBCshells","precomp","cutoff","homogen",
		      "filesval","filesvec"};
  enum {Ns,Nf,dof,L,n,levels,PBClevels,PBCshells,precomp,cutoff,homogen,
	filesval,filesvec};

  FILE *f;

  // Initializes the parameter values
  param->Ns = 10;
  param->Nf = 10;
  param->dof = 3;
  param->L    = 1;
  param->n    = 4;
  param->levels = 3;
  param->PBClevels  = 0;
  param->PBCshells = 0;
  param->precomp = 1;
  param->cutoff = 96;
  param->homogen = 1;
  //*param->filesval="t.out";
  //*param->filesvec="u.out";

  // Open options file for reading
  f = fopen(options,"r");

  // Reads one line at a time until EOF is reached
  while (fgets(line,80,f) != NULL) {
    // Read in tag
    sscanf(line,"%s",tag);

    // Determine the id corresponding to the tag
    tag_id = -1;
    for (i=0;i<ntags;i++) {
      if (strcmp(tag,alltags[i]) == 0) {
	tag_id = i;
	break;
      }
    }

    // Stores the selected parameter in the appropriate variable
    switch (tag_id) {
      // Read in number of sources
    case Ns:
      sscanf(line,"%s %d",tag,&param->Ns);
      break;

      // Read in number of field points
    case Nf:
      sscanf(line,"%s %d",tag,&param->Nf);
      break;

      // Read in number of degrees of freedom
    case dof:
      sscanf(line,"%s %d",tag,&param->dof);
      break;

      // Read in length of one side of simulation cube
    case L:
#ifdef SINGLE
      sscanf(line,"%s %f",tag,&param->L);
#else
      sscanf(line,"%s %lf",tag,&param->L);
#endif
      break;

      // Read in number of Chebyshev nodes in one direction
    case n:
      sscanf(line,"%s %d",tag,&param->n);
      break;

      // Read in maximum number of levels in octree
    case levels:
      sscanf(line,"%s %d",tag,&param->levels);
      break;

      // Read in number of PBC levels
    case PBClevels:
      sscanf(line,"%s %d",tag,&param->PBClevels);
      break;

      // Read in number of PBC shells
    case PBCshells:
      sscanf(line,"%s %d",tag,&param->PBCshells);
      break;

      // Read in pre-computation switch
    case precomp:
      sscanf(line,"%s %d",tag,&param->precomp);
      break;

      // Read in number of singular values to keep
    case cutoff:
      sscanf(line,"%s %d",tag,&param->cutoff);
      break;
      
      // Read in order of homogeneity of kernel
    case homogen:
#ifdef SINGLE
      sscanf(line,"%s %f",tag,&param->homogen);
#else
      sscanf(line,"%s %lf",tag,&param->homogen);
#endif
      break;

      // Read in file name for singular values
    case filesval:
      sscanf(line,"%s %s",tag,param->filesval);
      break;

      // Read in file name for singular vectors
    case filesvec:
      sscanf(line,"%s %s",tag,param->filesvec);
      break;

    default:
      printf("%s is an invalid tag!\n",tag);
    }
  }

  // Closes options file
  fclose(f);

  return 0;
}

#define frand_fast(xmin, xmax, inv_rand_max) ((xmin) + ((xmax) - (xmin)) * rand() * inv_rand_max )

/*
 * Function: SetSources
 * -------------------------------------------------------------------
 * Distributes an equal number of positive and negative charges uniformly
 * in the simulation cell ([-0.5*L,0.5*L])^3 and take these same locations
 * as the field points.
 */
//111206void SetSources(real3 *field, real3 *source, real *q, int N, int dof, real L)
void SetSources(real3 *field, real3 *source, real *q, int N, int dof, real L, int seed)
{
  /* Set a seed if necessary */
#if defined(RANDOM_SEED)
  srand(seed);
#endif

  /* Distributes the sources uniformly in the cubic cell */

#if (1) // slightly better, but not significant

  real halfp = L / 2;
  real halfm = - halfp;
  real inv_rand_max = 1.0 / RAND_MAX;

#if (DOF != 1)
  int k = 0;
#endif
  for (int i = 0; i < N; i ++) {

    real tmp = 1 - 2 * (i % 2); // 1 if i is even, -1 otherwise

#if (DOF == 1)
    q[i] = tmp;
#else
    for (int j = 0; j < dof; j ++) {
      q[k] = tmp;
      k ++;
    }
#endif
    source[i].x = frand_fast(halfm, halfp, inv_rand_max);
    source[i].y = frand_fast(halfm, halfp, inv_rand_max);
    source[i].z = frand_fast(halfm, halfp, inv_rand_max);
  }

#else

  int k = 0;
  for (int i = 0; i < N; i ++) {
    real tmp;
    if (i % 2 == 0)
      tmp = 1;
    else
      tmp = - 1;
    
    for (int j = 0; j < dof; j ++) {
      q[k] = tmp;
      k ++;
    }
    
    source[i].x = frand(- 0.5, 0.5) * L;
    source[i].y = frand(- 0.5, 0.5) * L;
    source[i].z = frand(- 0.5, 0.5) * L;
  }

#endif

  /* Takes the source locations as the field points */
#if defined(FIELDPOINTS_EQ_SOURCES)
#else
#error field must be defined here or somewhere else.
#endif
}

/*
 * Function: DirectCalc3D
 * ---------------------------------------------------------------------
 * Computes the potential at the first field point and returns 
 * the result in phi.
 */
void DirectCalc3D(real3 *field, real3 *source, real *q,
		  int Nf, int Ns, int dof, int start, real L, real *phi)
{

  /* Compute the interactions inside the computational cell */
  EvaluateField(field, source, q, Nf, Ns, dof, phi);

  /* Compute the interactions due to the periodic image cells */
  for (int i = 0; i < Nf; i ++) {
    real sum = 0;
    for (int l1 = - start; l1 < start + 1; l1 ++) {
      real3 cshift;
      cshift.x = (real)l1 * L;
      for (int l2 = - start; l2 < start + 1; l2 ++) {
	cshift.y = (real)l2 * L;
	for (int l3 = - start; l3 < start + 1; l3 ++) {
	  cshift.z = (real)l3 * L;
	  if (l1 != 0 || l2 != 0 || l3 != 0) {
	    for (int j = 0; j < Ns; j ++) {
	      real3 sourcepos;
	      sourcepos.x = source[j].x + cshift.x;
	      sourcepos.y = source[j].y + cshift.y;
	      sourcepos.z = source[j].z + cshift.z;
	      real Kij = EvaluateKernel(field[i], sourcepos);
	      sum += q[j] * Kij;
	    }
	  }
	}
      }
    }
    phi[i] = q[i] * (phi[i] + sum);
  }
  
} 

/*
 * Function: EwaldSolution
 * ---------------------------------------------------------------------
 * Computes the Ewald solution for 1/r, r.x/r^3, and 1/r^4 kernels
 * (for comparision to PBC with FMM).
 */
void EwaldSolution(real3 *field, real3 *source, real *q, int Nf, 
		   int Ns, int nstart, int mstart, real L, real beta,
		   real *phi, real *corr) {
  /*******************************
   *    LAPLACIAN KERNEL (1/r)   *
   *******************************/
#ifdef LAPLACIAN
  int i, j, m1, m2, m3, n1, n2, n3;
  real3 diff, cshift;
  real sum, tot, r, s, s2, f, g, mdotr;
  real pi = (real)M_PI;
  real beta2 = beta*beta;

  tot = 0;
  for (i=0;i<Nf;i++) {
    sum = 0;
    for (j=0;j<Ns;j++) {
      // Compute the direct contribution for n = [0 0 0]
      if (source[j].x != field[i].x || source[j].y != field[i].y ||
	  source[j].z != field[i].z) {
	diff.x = source[j].x - field[i].x;
	diff.y = source[j].y - field[i].y;
	diff.z = source[j].z - field[i].z;
	r = SQRT(diff.x*diff.x+diff.y*diff.y+diff.z*diff.z);
	
	s = beta*r;
	sum += q[j]*ERFC(s)/r;
      }

      // Compute the rest of the direct sum
      for (n1=-nstart;n1<nstart+1;n1++) {
	cshift.x = (real)n1*L;
	for (n2=-nstart;n2<nstart+1;n2++) {
	  cshift.y = (real)n2*L;
	  for (n3=-nstart;n3<nstart+1;n3++) {
	    cshift.z = (real)n3*L;
	    if (n1 != 0 || n2 != 0 || n3 != 0) {
	      diff.x = (source[j].x + cshift.x) - field[i].x;
	      diff.y = (source[j].y + cshift.y) - field[i].y;
	      diff.z = (source[j].z + cshift.z) - field[i].z;
	      r = SQRT(diff.x*diff.x+diff.y*diff.y+diff.z*diff.z);
	      
	      s = beta*r;
	      sum += q[j]*ERFC(s)/r;
	    }
	  }
	}
      }
    }

    // Add the direct contribution and subtract self-energy to the total energy
    tot += q[i]*(sum - 2*beta*q[i]/SQRT(pi));
  }

  sum = 0;
  // Compute the reciprocal sum
  for (m1=-mstart;m1<mstart+1;m1++) {
    for (m2=-mstart;m2<mstart+1;m2++) {
      for (m3=-mstart;m3<mstart+1;m3++) {
	if (m1 != 0 || m2 != 0 || m3 != 0) {
	  r = SQRT((real)(m1*m1+m2*m2+m3*m3));	      
	  s = pi*r/beta;
	  s2 = s*s;
	  f = 0;
	  g = 0;
	  for (j=0;j<Ns;j++) {
	    mdotr = (real)m1*source[j].x + (real)m2*source[j].y +
	      (real)m3*source[j].z;
	    f += q[j]*COS(2*pi*mdotr);
	    g += q[j]*SIN(2*pi*mdotr);
	  }
	  sum += 1/s2*(f*f+g*g)*EXP(-s2);
	}
      }
    }
  }

  // Add the reciprocal contribution to the total energy
  tot += sum*pi/beta2;

  // Return potential energy
  *phi = tot;

  // Compute the spherical shell extrinisic correction
  diff.x = 0;
  diff.y = 0;
  diff.z = 0;
  for (j=0;j<Ns;j++) {
    diff.x += q[j]*source[j].x;
    diff.y += q[j]*source[j].y;
    diff.z += q[j]*source[j].z;
  }

  // Return correction to the total energy (for comparision with direct sum)
  *corr = 4*pi/3*(diff.x*diff.x+diff.y*diff.y+diff.z*diff.z);


  /****************************************************
   * LAPLACIAN FORCE (r.x/r^3) - only the x-component *
   ****************************************************/
#elif LAPLACIANFORCE
  int i, j, m1, m2, m3, n1, n2, n3;
  real3 diff, cshift;
  real sum, r, r2, s, s2, f, g, mdotr;
  real pi = (real)M_PI;
  real fac1 = 2*beta/SQRT(pi), fac2;
  real *fi;
  fi = (real *)malloc(Nf * sizeof(real));
	
  for (i=0;i<Nf;i++) {
    sum = 0;
    for (j=0;j<Ns;j++) {
      // Compute force contribution from the n = [0 0 0] term of direct sum
      if (source[j].x != field[i].x || source[j].y != field[i].y ||
	  source[j].z != field[i].z) {
	diff.x = source[j].x - field[i].x;
	diff.y = source[j].y - field[i].y;
	diff.z = source[j].z - field[i].z;
	r = SQRT(diff.x*diff.x+diff.y*diff.y+diff.z*diff.z);
	r2 = r*r;
	s = beta*r;
	s2 = s*s;
	sum += q[j]*(fac1*EXP(-s2)+ERFC(s)/r)*diff.x/r2;
      }

      // Compute contribution from the rest of the direct sum
      for (n1=-nstart;n1<nstart+1;n1++) {
	cshift.x = (real)n1*L;
	for (n2=-nstart;n2<nstart+1;n2++) {
	  cshift.y = (real)n2*L;
	  for (n3=-nstart;n3<nstart+1;n3++) {
	    cshift.z = (real)n3*L;
	    if (n1 != 0 || n2 != 0 || n3 != 0) {
	      diff.x = (source[j].x + cshift.x) - field[i].x;
	      diff.y = (source[j].y + cshift.y) - field[i].y;
	      diff.z = (source[j].z + cshift.z) - field[i].z;
	      r = SQRT(diff.x*diff.x+diff.y*diff.y+diff.z*diff.z);
	      r2 = r*r;
	      s = beta*r;
	      s2 = s*s;
	      sum += q[j]*(fac1*EXP(-s2)+ERFC(s)/r)*diff.x/r2;
	    }
	  }
	}
      }
    }

    // Stores the force contribution from the direct sum
    fi[i] = sum;
  }

  // Compute the force contribution from the reciprocal sum
  sum = 0;
  for (m1=-mstart;m1<mstart+1;m1++) {
    for (m2=-mstart;m2<mstart+1;m2++) {
      for (m3=-mstart;m3<mstart+1;m3++) {
	if (m1 != 0 || m2 != 0 || m3 != 0) {
	  r = SQRT((real)(m1*m1+m2*m2+m3*m3));
	  r2 = r*r;
	  s = pi*r/beta;
	  s2 = s*s;
	  f = 0;
	  g = 0;
	  for (j=0;j<Ns;j++) {
	    mdotr = (real)m1*source[j].x + (real)m2*source[j].y +
	      (real)m3*source[j].z;
	    f += q[j]*COS(2*pi*mdotr);
	    g += q[j]*SIN(2*pi*mdotr);
	  }

	  fac2 = 2*(real)m1/r2*EXP(-s2);
	  for (i=0;i<Nf;i++) {
 	    mdotr = (real)m1*field[i].x + (real)m2*field[i].y +
	      (real)m3*field[i].z;
	    fi[i] += fac2*(g*COS(2*pi*mdotr)-f*SIN(2*pi*mdotr));
	  }
	}
      }
    }
  }

  // Return the force vector
  for (i=0;i<Nf;i++)
    phi[i] = q[i]*fi[i];

  free(fi);

  // Compute the force contribution from spherical shell extrinisic correction
  // First compute the x-component of the dipole
  diff.x = 0;
  for (j=0;j<Ns;j++) {
    diff.x += q[j]*source[j].x;
  }

  // Return correction to the force (for comparision with direct sum)
  *corr = 4*pi/3*diff.x;
  
  /**********************
   *    1/r^4 KERNEL    *
   **********************/
#elif ONEOVERR4
  int i, j, m1, m2, m3, n1, n2, n3;
  real3 diff, cshift;
  real sum, tot, r, s, s2, s4, f, g, mdotr;
  real pi = (real)M_PI;
  real fac = 2*pi*SQRT(pi)*beta;
  real beta2 = beta*beta;
  real beta4 = beta2*beta2;

  tot = 0;
  for (i=0;i<Nf;i++) {
    sum = 0;
    for (j=0;j<Ns;j++) {
      // Compute the direct contribution for n = [0 0 0]
      if (source[j].x != field[i].x || source[j].y != field[i].y ||
	  source[j].z != field[i].z) {
	diff.x = source[j].x - field[i].x;
	diff.y = source[j].y - field[i].y;
	diff.z = source[j].z - field[i].z;
	r = SQRT(diff.x*diff.x+diff.y*diff.y+diff.z*diff.z);
	
	s = beta*r;
	s2 = s*s;
	s4 = s2*s2;
	g = EXP(-s2)*(s2+1);
	sum += q[j]*g/s4;
      }

      // Compute the rest of the direct sum
      for (n1=-nstart;n1<nstart+1;n1++) {
	cshift.x = (real)n1*L;
	for (n2=-nstart;n2<nstart+1;n2++) {
	  cshift.y = (real)n2*L;
	  for (n3=-nstart;n3<nstart+1;n3++) {
	    cshift.z = (real)n3*L;
	    if (n1 != 0 || n2 != 0 || n3 != 0) {
	      diff.x = (source[j].x + cshift.x) - field[i].x;
	      diff.y = (source[j].y + cshift.y) - field[i].y;
	      diff.z = (source[j].z + cshift.z) - field[i].z;
	      r = SQRT(diff.x*diff.x+diff.y*diff.y+diff.z*diff.z);
	      
	      s = beta*r;
	      s2 = s*s;
	      s4 = s2*s2;
	      g = EXP(-s2)*(s2+1);
	      sum += q[j]*g/s4;
	    }
	  }
	}
      }

      // Compute the reciprocal sum
      diff.x = source[j].x - field[i].x;
      diff.y = source[j].y - field[i].y;
      diff.z = source[j].z - field[i].z;
      for (m1=-mstart;m1<mstart+1;m1++) {
	for (m2=-mstart;m2<mstart+1;m2++) {
	  for (m3=-mstart;m3<mstart+1;m3++) {
	    r = SQRT((real)(m1*m1+m2*m2+m3*m3));
	      
	    s = pi*r/beta;
	    s2 = s*s;
	    f = fac*(EXP(-s2)-SQRT(pi)*s*ERFC(s));
	    mdotr = (real)m1*diff.x + (real)m2*diff.y + (real)m3*diff.z;
	    sum += q[j]*f*COS(-2*pi*mdotr);
	  }
	}
      }
    }

    // Add to total potential energy and subtract self-energy
    tot += q[i]*(sum - (real)0.5*beta4*q[i]);
  }

  // Return the potential energy and correction
  *phi = tot;
  *corr = 0;

  #endif
} 

/*
 * Function: EvaluateKernel
 * -------------------------------------------------------------------
 * Evaluates the kernel given a source and a field point.
 */
real EvaluateKernel(real3 fieldpos, real3 sourcepos)
{
  /*********************************
   *    LAPLACIAN KERNEL (1/r)     *
   *********************************/
#ifdef LAPLACIAN
  real3 diff;
  real rinv;

  // Compute 1/r
  diff.x = sourcepos.x - fieldpos.x;
  diff.y = sourcepos.y - fieldpos.y;
  diff.z = sourcepos.z - fieldpos.z;
  rinv   = ONE/SQRT(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
  
  // Output result (1/r)
  return rinv;

  /****************************************************
   * LAPLACIAN FORCE (r.x/r^3) - only the x-component *
   ****************************************************/
#elif LAPLACIANFORCE
  real3 diff;
  real rinv;

  // Compute 1/r
  diff.x = sourcepos.x - fieldpos.x;
  diff.y = sourcepos.y - fieldpos.y;
  diff.z = sourcepos.z - fieldpos.z;
  rinv   = ONE/SQRT(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
  
  // Output result (r.x/r^3)
  return diff.x*rinv*rinv*rinv;

  /*********************
   *    1/r^4 KERNEL   *
   *********************/
#elif ONEOVERR4
  real3 diff;
  real rinv;

  // Compute 1/r
  diff.x = sourcepos.x - fieldpos.x;
  diff.y = sourcepos.y - fieldpos.y;
  diff.z = sourcepos.z - fieldpos.z;
  rinv   = ONE/SQRT(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
  
  // Output result (1/r^4)
  return rinv*rinv*rinv*rinv;

#endif
}


/*
 * Function: EvaluateKernelCell
 * -------------------------------------------------------------------
 * Evaluates the kernel for interactions between a pair of cells.
 */
void EvaluateKernelCell(real3 *field, real3 *source, int Nf, 
			int Ns, int dof, real *kernel) {
  int i, j, k, count;
  int dofNf = dof*Nf;
  real Kij;

  count = 0;
  for (j=0;j<Ns;j++) {
    for (i=0;i<Nf;i++) {
      Kij = EvaluateKernel(field[i],source[j]);
      for (k=0;k<dof;k++)
	kernel[count+k*(dofNf+1)] = Kij; 
      count += dof;
    }
    count += (dof-1)*(dofNf);
  }
}

/*
 * Function: EvaluateField
 * -------------------------------------------------------------------
 * Evaluates the field due to interactions between a pair of cells.
 */
void EvaluateField(real3 *field, real3 *source, real *q, int Nf, 
		   int Ns, int dof, real *fieldval) {
  
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < Nf; i ++) {

    real *kernel = (real *)malloc(Ns * sizeof(real));

    /* Compute the interaction between each field point and source */
    for (int j = 0; j < Ns; j ++) {
      if (source[j].x != field[i].x || source[j].y != field[i].y || source[j].z != field[i].z) {
	kernel[j] = EvaluateKernel(field[i], source[j]);
      }
      else {
	kernel[j] = ZERO;
      }
    }	

    int l = dof * i;

    /* Sum over all sources */
    for (int k = 0; k < dof; k ++) {
      int incr = 1;
      fieldval[l] = adot_(&Ns, &q[k], &dof, kernel, &incr);
      l ++;
    }

    free(kernel);

  }

}

/*
 * Function: ComputeWeights
 * ------------------------------------------------------------------
 * Computes the weights for the Chebyshev nodes of all children cells
 * (identical for all cells and all levels so just compute once and
 * store in memory) and set up the lookup table.
 */
void ComputeWeights(real *Tkz, int *Ktable, real *Kweights, real *Cweights, int n) 
{
  real *nodes = (real *)malloc(n * sizeof(real));
  real *vec = (real *)malloc(n * sizeof(real));
  int n3 = n * n * n;
  int Nc = 2 * n3; // Number of child Chebyshev nodes
  real3 *fieldt = (real3 *)malloc(Nc * sizeof(real3)); // Chebyshev-transformed coordinates
  real3 *Sn = (real3 *)malloc(n * Nc * sizeof(real3));
  real pi = (real)M_PI; 
  
  /* Initialize lookup table */
  for (int i = 0; i < 343; i ++) {
    Ktable[i] = - 1;
  }
  
  /* Create lookup table */
  int ncell = 0;
  int ninteract = 0;
  for (int l1 = - 3; l1 < 4; l1 ++) {
    for (int l2 = - 3; l2 < 4; l2 ++) {
      for (int l3 = - 3; l3 < 4; l3 ++) {
	if (abs(l1) > 1 || abs(l2) > 1 || abs(l3) > 1) {
	  Ktable[ncell] = ninteract;
	  ninteract++;
	}
	ncell++;
      }
    }
  }	

  /* Compute the n Chebyshev nodes of T_n(x) */
  for (int m = 0; m < n; m ++) {
    nodes[m] = COS(pi * (real)(2 * m + 1) / (real)(2 * n));
  }

  /* Evaluate the Chebyshev polynomials of degree 0 to n-1 at the nodes */
  for (int m = 0; m < n; m ++) {
    ComputeTk(nodes[m], n, vec);
    int i = m * n;
    for (int k = 0; k < n; k ++) {
      Tkz[i + k] = vec[k];
    }
  }

  /* Compute the weights for the kernel matrix K */
  int count = 0;
  for (int l1 = 0; l1 < n; l1 ++) {
    real tmp1 = 1 / SQRT(1 - nodes[l1] * nodes[l1]);
    for (int l2 = 0; l2 < n; l2 ++) {
      real tmp2 = tmp1 / SQRT(1 - nodes[l2] * nodes[l2]);
      for (int l3 = 0; l3 < n; l3 ++) {
	Kweights[count] = tmp2 / SQRT(1 - nodes[l3] * nodes[l3]);
	count ++;
      }
    }
  }
   
  /* Map all Chebyshev nodes from the children cells to the parent */
  int k = 0;
  for (int i = 0; i < 2; i ++) {

    /* Determine the mapping function for the specific child cell */
    real3 vtmp;
    vtmp.x = - 1;
    vtmp.y = - 1;
    if (i == 0) {
      vtmp.z = - 1;
    } else {
      vtmp.z = 1;
    }
    
    for (int l1 = 0; l1 < n; l1 ++) {
      for (int l2 = 0; l2 < n; l2 ++) {
	for (int l3 = 0; l3 < n; l3 ++) {
	  fieldt[k].x = (real)0.5 * (nodes[l1] + vtmp.x);
	  fieldt[k].y = (real)0.5 * (nodes[l2] + vtmp.y);
	  fieldt[k].z = (real)0.5 * (nodes[l3] + vtmp.z);
	  k ++;
	}
      }
    }
  }
    
  /* Compute Sc, the mapping function for the field points */
  ComputeSn(fieldt, Tkz, n, Nc, Sn);

  /* Extract out the Chebyshev weights */
  count = 0;
  for (int l1 = 0; l1 < n; l1 ++) {
    int k = l1 * Nc;
    for (int l2 = 0; l2 < n; l2 ++) {
      Cweights[count] = Sn[k + l2].z;
      count ++;
    }
  }
  for (int l1 = 0; l1 < n; l1 ++) {
    int k = l1 * Nc;
    for (int l2 = 0; l2 < n; l2 ++) {
      Cweights[count] = Sn[k + n3 + l2].z;
      count ++;
    }
  }
  free(nodes);
  free(vec);
  free(fieldt);
  free(Sn);
}

/*
 * Function: ComputeWeightsPBC
 * ------------------------------------------------------------------
 * Computes the weights for the Chebyshev nodes of all children cells
 * (identical for all cells and all levels so just compute once and
 * store in memory) (for PBC calculation each cell has 27 children
 * instead of 8).
 */
void ComputeWeightsPBC(real *Wup, int n, int dof) {
  int i, j, k, l, m, l1, l2, l3, count1, count2, count3;
  real3 vtmp;

  real *nodes, *vec, *Tkz;
  real *Sxyz, *W;
  real3 *fieldt, *Sn;
 
  nodes = (real *)malloc(n * sizeof(real));
  vec   = (real *)malloc(n * sizeof(real));
  Tkz   = (real *)malloc(n * n * sizeof(real));
 
  int n3 = n*n*n;                     // n3 = n^3
  int n6 = n3*n3;                     // n6 = n^6
  int dofn3 = dof*n3;
  int dof2n3 = dof*dofn3;
  real prefac = 2/(real)n;
  real prefac3 = prefac*prefac*prefac;
  int Nc = 27*n3;                     // Number of child Chebyshev nodes
  fieldt = (real3 *)malloc(Nc * sizeof(real3)); // Chebyshev-transformed coordinates
  Sn     = (real3 *)malloc(n * Nc * sizeof(real3));
  Sxyz   = (real *)malloc(n3 * Nc * sizeof(real));
  W      = (real *)malloc(n6 * sizeof(real));
  real pi = (real)M_PI, fac = ONE / THREE, tmp;

  char filename[]="weightpbcu.out";
  FILE *f;

  // Compute the n Chebyshev nodes of T_n(x)
  for (m=0;m<n;m++)
    nodes[m] = COS(pi*(real)(2*m+1)/(real)(2*n));

  // Evaluate the Chebyshev polynomials of degree 0 to n-1 at the nodes
  for (m=0;m<n;m++) {
    ComputeTk(nodes[m],n,vec);
    i = m*n;
    for (k=0;k<n;k++)
      Tkz[i+k] = vec[k];
  }
   
  // Map all Chebyshev nodes from the children cells to the parent
  k = 0;
  for (i=0;i<27;i++) {
    // Determine the mapping function for the specific child cell
    if (i<9) {
      vtmp.x = -2;
      
      if (i<3)
	vtmp.y = -2;
      else if (i<6)
	vtmp.y = 0;
      else
	vtmp.y = 2;
	
    } else if (i<18) {
      vtmp.x = 0;
      
      if (i<12)
	vtmp.y = -2;
      else if (i<15)
	vtmp.y = 0;
      else
	vtmp.y = 2;

    } else {
      vtmp.x = 2;
      
      if (i<21)
	vtmp.y = -2;
      else if (i<24)
	vtmp.y = 0;
      else
	vtmp.y = 2;
    }
      
    if (i%3 == 0)
      vtmp.z = -2;
    else if (i%3 == 1)
      vtmp.z = 0;
    else
      vtmp.z = 2;
    
    for (l1=0;l1<n;l1++) {
      for (l2=0;l2<n;l2++) {
	for (l3=0;l3<n;l3++) {
	  fieldt[k].x = fac*(nodes[l1] + vtmp.x);
	  fieldt[k].y = fac*(nodes[l2] + vtmp.y);
	  fieldt[k].z = fac*(nodes[l3] + vtmp.z);
	  k++;
	}
      }
    }
  }
    
  // Compute Sc, the mapping function for the field points
  ComputeSn(fieldt,Tkz,n,Nc,Sn);

  // Compute Sxyz, the weights for the sources
  for (k=0;k<Nc;k++) {
    l = 0;
    for (l1=0;l1<n;l1++) {
      for (l2=0;l2<n;l2++) {
	for (l3=0;l3<n;l3++) {
	  Sxyz[l*Nc+k] = Sn[l1*Nc+k].x*Sn[l2*Nc+k].y*Sn[l3*Nc+k].z;
	  l++;
	}
      }
    }
  }

  // Accumulate the weights into a single weight matrix W
  for (i=0;i<n6;i++)
    W[i] = 0;

  for (i=0;i<27;i++) {
    count1 = 0;
    count2 = i*n3;
    for (j=0;j<n3;j++) {
      for (k=0;k<n3;k++) {
	W[count1] += 0;///Sxyz[k*Nc+count2];
	count1++;
      }
      count2++;
    }
  }

  // Compute Wup
  count1 = 0;
  for (i=0;i<n3;i++) {
    count2 = i*dof2n3;
    for (j=0;j<n3;j++) {
      tmp = prefac3*W[count1];
      for (k=0;k<dof;k++) {
	count3 = count2 + k*dofn3;
	for (l=0;l<dof;l++) {
	  Wup[count3] = tmp;
	  count3++;
	}
      }
      count1++;
      count2 += dof;
    }
  }
 
  // Output Wup to file
  f = fopen(filename,"w");
  count1 = 0;
  for (i=0;i<n3;i++) {
    for (j=0;j<n3;j++) {
      fprintf(f,"%18.12e ",Wup[count1]);
      count1++;
    }
    fprintf(f,"\n");
  }
  fclose(f); 
  free(nodes);
  free(vec);
  free(Tkz);
  free(fieldt);
  free(Sn);
  free(Sxyz);
  free(W);
   
}

/*
 * Function: ComputeKernelSVD
 * ---------------------------------------------------------------------
 * Computes the kernel for 316n^6 interactions between Chebyshev nodes
 * and then computes the SVD of the kernel matrix.
 */
void ComputeKernelSVD(real *Kweights, int n, int dof, char *Kmat,
		      char *Umat) {
  int i, j, l, m, k1, k2, k3, l1, l2, l3, m1, m2, m3, z;
  int count, count1, count2, count3;
  real3 scenter, vtmp;
  real tmp, sweight, fweight;

  int n3 = n*n*n;            // n3 = n^3
  int dofn3 = dof*n3;
  int dof2n6 = dofn3*dofn3;
  int cols = 316*dofn3;
  int cutoff;

  real *K0, *U0, *Sigma, *VT=NULL;
  real *nodes, *kernel, *work;
  real3 *fieldpos, *sourcepos;

  K0 = (real *)malloc(316 * dof2n6 * sizeof(real));
  U0 = (real *)malloc(dof2n6 * sizeof(real));
  Sigma = (real *)malloc(dofn3 * sizeof(real));
  nodes  = (real *)malloc(n * sizeof(real));
  kernel = (real *)malloc(dof2n6 * sizeof(real));
  fieldpos = (real3 *)malloc(n3 * sizeof(real3));
  sourcepos = (real3 *)malloc(n3 * sizeof(real3));


  real *Kcell[316];

  for (z = 0; z < 316; ++z)
     Kcell[z] = (real *)malloc(dof2n6 * sizeof(real));

  real pi = (real)M_PI;

  char save[]="S", nosave[]="N";
  int nosavedim=1;
  int info, lwork=3*cols;
  work = (real *)malloc(lwork * sizeof(real));
  FILE *f;

  /* Compute the n Chebyshev nodes of T_n(x) */
  for (m=0;m<n;m++)
    nodes[m] = COS(pi*(real)(2*m+1)/(real)(2*n));

  /* Compute the locations of the field points */
  count = 0;
  for (l1=0;l1<n;l1++) {
    vtmp.x = (real)0.5*nodes[l1];
    for (l2=0;l2<n;l2++) {
      vtmp.y = (real)0.5*nodes[l2];
      for (l3=0;l3<n;l3++) {
	fieldpos[count].x = vtmp.x;
	fieldpos[count].y = vtmp.y;
	fieldpos[count].z = (real)0.5*nodes[l3];
	count++;
      }
    }
  }

  /* Compute the kernel values for interactions with all 316 cells */
  count = 0;
  for (k1=-3;k1<4;k1++) {
    scenter.x = (real)k1;
    for (k2=-3;k2<4;k2++) {
      scenter.y = (real)k2;
      for (k3=-3;k3<4;k3++) {
	scenter.z = (real)k3;
	if (abs(k1) > 1 || abs(k2) > 1 || abs(k3) > 1) {
	  count1 = 0;
	  for (l1=0;l1<n;l1++) {
	    vtmp.x = scenter.x + (real)0.5*nodes[l1];
	    for (l2=0;l2<n;l2++) {
	      vtmp.y = scenter.y + (real)0.5*nodes[l2];
	      for (l3=0;l3<n;l3++) {
		sourcepos[count1].x = vtmp.x;
		sourcepos[count1].y = vtmp.y;
		sourcepos[count1].z = scenter.z + (real)0.5*nodes[l3];
		count1++;
	      }
	    }
	  }
	 
	  /* Compute the kernel at each of the field Chebyshev nodes */
	  EvaluateKernelCell(fieldpos,sourcepos,n3,n3,dof,kernel);

	  /* Copy the kernel values to the appropriate location */
	  count1 = 0;
	  count2 = 0;
	  for (l1=0;l1<n;l1++) {
	    for (l2=0;l2<n;l2++) {
	      for (l3=0;l3<n;l3++) {
		sweight = Kweights[count2];
		for (l=0;l<dof;l++) {
		  count3 = 0;
		  for (m1=0;m1<n;m1++) {
		    for (m2=0;m2<n;m2++) {
		      for (m3=0;m3<n;m3++) { 
			fweight = Kweights[count3];
			for (m=0;m<dof;m++) {
			  tmp = kernel[count1]/(sweight*fweight);
			  K0[count] = tmp;
			  count++;
			  count1++;
			}
			count3++;
		      }
		    }
		  }
		}
		count2++;
	      }
	    }
	  }
	}
      }
    }
  }

  /* Extract the submatrix for each of the 316 cells */
  count = 0;
  for (i=0;i<316;i++) {
    for (j=0;j<dof2n6;j++) {
      Kcell[i][j] = K0[count];
      count++;
    }
  }
 
  /* Compute the SVD of K0 */
  agesvd_(save,nosave,&dofn3,&cols,K0,&dofn3,Sigma,U0,&dofn3,VT,&nosavedim,
	  work,&lwork,&info);

  /* Determine the number of singular values to keep */
  cutoff = n3/2 - 1;
#ifndef DISABLE_INCREASE_CUTOFF_BY_ONE
  cutoff++;
#endif

  /* Include all singular values of the same magnitude */
#ifdef ENABLE_MODIFY_CUTOFF
  while (fabs(Sigma[cutoff+1]-Sigma[cutoff])/fabs(Sigma[0]) < 1.e-9)
    cutoff++;
#endif

  int cutoff2 = cutoff*cutoff;
  real *KV, *Ecell, *U;
  KV    = (real *)malloc(dofn3 * cutoff * sizeof(real));
  Ecell = (real *)malloc(cutoff2 * sizeof(real));
  U     = (real *)malloc(cutoff * dofn3 * sizeof(real));

  char *transa, *transb;
  real alpha=1, beta=0;

  /* Extract the needed columns from U0 and write out to a file */
  f = fopen(Umat,"w");
#ifdef ENABLE_ASCII_IO
  fprintf(f,"%d\n",cutoff);    // Output the cutoff value
#else
  fwrite(&cutoff, sizeof(int), 1, f);    // Output the cutoff value
#endif
  count = dofn3*cutoff;
#ifdef ENABLE_ASCII_IO
  for (i=0;i<count;i++) {
    U[i] = U0[i];
    fprintf(f,"%18.12e\n",U0[i]);
  }
#else
  for (i = 0; i < count; i++) {
    U[i] = U0[i];
  }
  fwrite(U0, sizeof(real), count, f);
#endif
  fclose(f);

  /* Compute E_{ijk} = U^T K_{ijk} V and write out to a file */
  f = fopen(Kmat,"w");
  count = 0;
  for (i=0;i<316;i++) {       
    /* Compute K_{ijk} V (replace U with V if kernel is non-symmetric) */
    transa = "n";
    transb = "n";
    agemm_(transa,transb,&dofn3,&cutoff,&dofn3,&alpha,Kcell[i],&dofn3,
	   U,&dofn3,&beta,KV,&dofn3);

    /* Then compute E_{ijk} */
    transa = "t";
    transb = "n";
    agemm_(transa,transb,&cutoff,&cutoff,&dofn3,&alpha,U,&dofn3,KV,&dofn3,
	   &beta,Ecell,&cutoff);

    /* Write out submatrix E_{ijk} */
#ifdef ENABLE_ASCII_IO
    for (j=0;j<cutoff2;j++) {
      fprintf(f,"%18.12e\n",Ecell[j]);
    }
#else
    fwrite(Ecell, sizeof(real), cutoff2, f);
#endif
  }
  fclose(f);
  free(K0);
  free(U0);
  free(Sigma);
  free(nodes);
  free(kernel);
  free(fieldpos);
  free(sourcepos);
  free(work);
  free(KV);
  free(Ecell);
  free(U);
 
  for (z = 0; z < 316; ++z)
     free(Kcell[z]);
}

/*
 * Function: ComputePeriodicKernel
 * ---------------------------------------------------------------------
 * Forms the matrix that describes the interactions of the computational
 * cell with its periodic images up to lpbc shells.
 */
void ComputePeriodicKernel(real *KPBC, real *Wup, real L, int n, 
			   int dof, int lpbc) {
  int i, j, l, l1, l2, l3, m, count, start;
  real *sval, *Srow;
  real Kij1, Kij2, scale, sum, corr;
  real3 sourcepos;

  int n3 = n*n*n;                          // n3 = n^3
  int dofn3 = dof*n3;
  real prefac   = 2/(real)n;         // Prefactor for Sn
  real prefac3  = prefac*prefac*prefac;  // prefac3 = prefac^3
  real nodes[n], vec[n], Tkz[n*n], kernel[dofn3][dofn3], pi = (real)M_PI;
  real sourceval[lpbc+1][dofn3];
  real3 nodepos[n3], cshift, vtmp, Sn[729*n], sourcet[729];

  char trans[]="n";
  real alpha=1, beta=0;
  int incr=1;

  // Compute the n Chebyshev nodes of T_n(x)
  for (m=0;m<n;m++)
    nodes[m] = COS(pi*(real)(2*m+1)/(real)(2*n));

  // Evaluate the Chebyshev polynomials of degree 0 to n-1 at the nodes
  for (m=0;m<n;m++) {
    ComputeTk(nodes[m],n,vec);
    i = m*n;
    for (l=0;l<n;l++)
      Tkz[i+l] = vec[l];
  }

  // Compute the locations of the field points (need to fix to handle dofn3)
  count = 0;
  for (l1=0;l1<n;l1++) {
    vtmp.x = (real)0.5*nodes[l1]*L;
    for (l2=0;l2<n;l2++) {
      vtmp.y = (real)0.5*nodes[l2]*L;
      for (l3=0;l3<n;l3++) {
	nodepos[count].x = vtmp.x;
	nodepos[count].y = vtmp.y;
	nodepos[count].z = (real)0.5*nodes[l3]*L;
	count++;
      }
    }
  }

  // Compute the field values due to nearby periodic sources
  if (lpbc == 0)
    start = 1;
  else if (lpbc == 1)
    start = 4;
  else
    start = 13;

  for (i=0;i<dofn3;i++) {
    for (j=0;j<dofn3;j++) {
      sum = 0;
      for (l1=-start;l1<start+1;l1++) {
	cshift.x = (real)l1*L;
	for (l2=-start;l2<start+1;l2++) {
	  cshift.y = (real)l2*L;
	  for (l3=-start;l3<start+1;l3++) {
	    cshift.z = (real)l3*L;
	    if (abs(l1) > 1 || abs(l2) > 1 || abs(l3) > 1) {
	      sourcepos.x = cshift.x + nodepos[j].x;
	      sourcepos.y = cshift.y + nodepos[j].y;
	      sourcepos.z = cshift.z + nodepos[j].z;
	      Kij1 = EvaluateKernel(nodepos[i],sourcepos);

	      // Add opposite charge if necessary
	      #ifdef LAPLACIAN
	      Kij2 = EvaluateKernel(nodepos[i],cshift);
	      #else
	      Kij2 = 0;
	      #endif

	      sum += (Kij1-Kij2);
	    }
	  }
	}
      }
      
      // Store the field value
      kernel[i][j] = sum;
     }
  }

  /* Compute initial set of pseudo-charges at Chebyshev nodes */    
  // Map all of the source points to the box ([-1 1])^3
  count = 0;
  for (l1=-4;l1<5;l1++) {
    vtmp.x = (real)2/(real)9*(real)l1;
    for (l2=-4;l2<5;l2++) {
      vtmp.y = (real)2/(real)9*(real)l2;
      for (l3=-4;l3<5;l3++) {
	sourcet[count].x = vtmp.x;
	sourcet[count].y = vtmp.y;
	sourcet[count].z = (real)2/(real)9*(real)l3;
	count++;
      }
    }
  }
    
  // Compute Ss, the mapping function for the sources
  ComputeSn(sourcet,Tkz,n,729,Sn);
  
  // Compute the source values
  if (lpbc > 2) {
    sval = sourceval[3];
    count = 0;
    for (l1=0;l1<n;l1++) {
      for (l2=0;l2<n;l2++) {
	for (l3=0;l3<n;l3++) {
	  sum = 0;
	  for (j=0;j<729;j++) {
	    sum += Sn[l1*729+j].x*Sn[l2*729+j].y*Sn[l3*729+j].z;
	  }
	  sval[count] = prefac3*sum; 
	  count++;
	}
      }
    }
  }

  // Compute Qklm, the pseudo-charges at each Chebyshev node
 
  for (l=3;l<lpbc;l++) {
    sval = sourceval[l];
    Srow = sourceval[l+1];
    agemv_(trans,&dofn3,&dofn3,&alpha,Wup,&dofn3,sval,&incr,&beta,
	   Srow,&incr);
  }

  // Compute the effect of periodic sources one shell at a time
  scale = 9.0;
  for (l=3;l<lpbc+1;l++) {
    sval = sourceval[l];

    // Compute the field values due to periodic sources in shell l
    for (i=0;i<dofn3;i++) {
      for (j=0;j<dofn3;j++) {
	sum = 0;
	for (l1=-4;l1<5;l1++) {
	  cshift.x = (real)l1*L;
	  for (l2=-4;l2<5;l2++) {
	    cshift.y = (real)l2*L;
	    for (l3=-4;l3<5;l3++) {
	      cshift.z = (real)l3*L;
	      if (abs(l1) > 1 || abs(l2) > 1 || abs(l3) > 1) {			
		for (m=0;m<dofn3;m++) {
		  sourcepos.x = (cshift.x+nodepos[m].x)*scale + nodepos[j].x; 
		  sourcepos.y = (cshift.y+nodepos[m].y)*scale + nodepos[j].y; 
		  sourcepos.z = (cshift.z+nodepos[m].z)*scale + nodepos[j].z; 
		  Kij1 = EvaluateKernel(nodepos[i],sourcepos);
		  
		  // Add opposite charge if necessary
		  #ifdef LAPLACIAN
		  sourcepos.x = (cshift.x+nodepos[m].x)*scale;
		  sourcepos.y = (cshift.y+nodepos[m].y)*scale;
		  sourcepos.z = (cshift.z+nodepos[m].z)*scale;
		  Kij2 = EvaluateKernel(nodepos[i],sourcepos);
		  #else
		  Kij2 = 0;
		  #endif

		  sum += sval[m]*(Kij1-Kij2);
		}
	      }
	    }
	  }
	}
	
	// Add to the field value 
	kernel[i][j] += sum;
      }
    }

    // Increase scaling factor
    scale *= 3.0;
  }
  
  // Output the periodic kernel matrix in column-major format
  count = 0;
  for (j=0;j<dofn3;j++) {
    corr = 4*pi/3*nodepos[j].x;
    for (i=0;i<dofn3;i++) {
      KPBC[count] = kernel[i][j] - corr;
      count++;
    }
  }
}  

/*
 * Function: ComputeTk
 * ------------------------------------------------------------------
 * Computes T_k(x) for k between 0 and n-1 inclusive.
 */
void ComputeTk(real x, int n, real *vec)
{
  vec[0] = 1;
  vec[1] = x;
  for (int k = 2; k < n; k ++) {
    vec[k] = 2 * x * vec[k - 1] - vec[k - 2];
  }
}

/*
 * Function: ComputeSn
 * ------------------------------------------------------------------
 * Computes S_n(x_m,x_i) for all Chebyshev node-point pairs using 
 * Clenshaw's recurrence relation.
 */
void ComputeSn(real3 *point, real *Tkz, int n, int N, real3 *Sn)
{
  real vec[n], d[n + 2];

  for (int m = 0; m < n; m ++) {
    /* Extract T_k for the Chebyshev node x_m */
    int k = m * n;
    for (int j = 0; j < n; j ++) {
      vec[j] = Tkz[k + j];
    }

    /* Compute S_n for each direction independently using Clenshaw */
    k = m * N;
    for (int i = 0; i < N; i ++) {

      real x = point[i].x;
      d[n] = d[n + 1] = 0;
      for (int j = n - 1; j > 0; j --) {
	d[j] = 2 * x * d[j + 1] - d[j + 2] + vec[j];
      }
      Sn[k + i].x = x * d[1] - d[2] + (real)0.5 * vec[0];

      real y = point[i].y;
      d[n] = d[n + 1] = 0;
      for (int j = n - 1; j > 0; j --) {
	d[j] = 2 * y * d[j + 1] - d[j + 2] + vec[j];
      }
      Sn[k + i].y = y * d[1] - d[2] + (real)0.5 * vec[0];

      real z = point[i].z;
      d[n] = d[n + 1] = 0;
      for (int j = n - 1; j > 0;j --) {
	d[j] = 2 * z * d[j + 1] - d[j + 2] + vec[j];
      }
      Sn[k + i].z = z * d[1] - d[2] + (real)0.5 * vec[0];

    }
  }
}


static void ComputeSn2each(const real *vec, real *d, const int n, const real x, real *Sn)
{
  d[n] = d[n + 1] = 0;
  for (int j = n - 1; j > 0; j --) {
    d[j] = 2 * x * d[j + 1] - d[j + 2] + vec[j];
  }
  *Sn = x * d[1] - d[2] + (real)0.5 * vec[0]; // Sn[k+i]=x*d[1]-d[2]+vec[0]/2
}


void ComputeSn2(real3 *point, real *Tkz, int n, int N, real *Snx, real *Sny, real *Snz)
{
#if defined(SLOW)
  real vec[n], d[n + 2];
#else
  real d[n + 2];
#endif

  for (int m = 0; m < n; m ++) {

    /* Extract T_k for the Chebyshev node x_m */
#if defined(SLOW)
    int k = m * n;
    for (int j = 0; j < n; j ++) { // LOOP WAS VECTORIZED.
      vec[j] = Tkz[k + j];
    }
#else
    const int mn = m * n;
    const real *vec = &(Tkz[mn]);
#endif

    /* Compute S_n for each direction independently using Clenshaw */
#if defined(SLOW)
    k = m * N;
    for (int i = 0; i < N; i ++) {
      real x = point[i].x;
      ComputeSn2each(vec, d, n, x, &(Snx[k + i]));
      real y = point[i].y;
      ComputeSn2each(vec, d, n, y, &(Sny[k + i]));
      real z = point[i].z;
      ComputeSn2each(vec, d, n, z, &(Snz[k + i]));
    }
#else
    const int mN = m * N;
    for (int i = 0; i < N; i ++) {
      ComputeSn2each(vec, d, n, point[i].x, &(Snx[mN + i]));
      ComputeSn2each(vec, d, n, point[i].y, &(Sny[mN + i]));
      ComputeSn2each(vec, d, n, point[i].z, &(Snz[mN + i]));
    }
#endif
  }
}
