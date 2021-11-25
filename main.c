/*
 * Function: main.c
 * Description: Provides an example of using the black-box fast multipole
 * method for computing fields in the simulation cell.
 * Usage: ./bbfmm N L n
 * ----------------------------------------------------------------------
 * 
 * Black-Box Fast Multipole Method (BBFMM)
 * William Fong
 * Stanford University
 *
 */

#include "bbfmm.h"

int main(int argc, char *argv[])
{
  /* Start main timer */
#ifndef DISABLE_TIMING
  allocTimer(&timer_main);
  initTimer(timer_main);
  startTimer(timer_main);
#endif

  /* Start running timer */
#ifndef DISABLE_TIMING
  allocTimer(&timer_run);
  initTimer(timer_run);
  startTimer(timer_run);
#endif
  
  /* Some setups */
#ifndef DISABLE_TIMING
  allocTimer(&timer_setup);
  initTimer(timer_setup);
  startTimer(timer_setup);
#endif

#ifndef DISABLE_TIMING
  allocTimer(&timer_comp); initTimer(timer_comp);
  allocTimer(&timer_upward); initTimer(timer_upward);
  allocTimer(&timer_interact); initTimer(timer_interact);
  allocTimer(&timer_downward); initTimer(timer_downward);
  allocTimer(&timer_output); initTimer(timer_output);
#endif

  if (argc < 4) {
    MESG("usage: ./bbfmm [-d l -s seed] N L n\n");
    exit(EXIT_FAILURE);
  }

  int l = 0; // initialise the number of levels
  int seed;
  bbfmm_options(argc, argv, &l, &seed); // optind=1 if no option is given

  int N = atoi(argv[optind]);            // Number of points
  real L = (real)atof(argv[optind + 1]); // Length of simulation cell (assumed to be a cube)
  int n = atoi(argv[optind + 2]);       // Number of Chebyshev nodes per dimension

  int dof  = DOF;           // Number of degrees of freedom (scalar = 1)
#if defined(FIELDPOINTS_EQ_SOURCES)
  int Ns   = N;             // Number of sources in simulation cell
  int Nf   = N;             // Number of field points in simulation cell
#else
#error Not implemented yet.
#endif

  real3 *field = (real3 *)malloc(Nf * sizeof(real3)); // Position array for the source points
#if defined(FIELDPOINTS_EQ_SOURCES)
  real3 *source = field; // link only; Position array for the source points
#else
  real3 *source = (real3 *)malloc(Ns * sizeof(real3));
#endif
  real *q = (real *)malloc(Ns * dof * sizeof(real)); // Source strength array
  real *phi = (real *)malloc(Nf * dof * sizeof(real)); // field value array

  /* Print arguments */
  INFO("N = %d\n", N);
  INFO("L = %f\n", L);
  INFO("n = %d\n", n);

  /* Print options defined by -D flags */
  opts();

  /* Print related environment variables */
  envs();

  /* Initialize the field points, source points, and their
     corresponding charges */
#ifndef DISABLE_TIMING
  timerType *timer_setup_particles;
  allocTimer(&timer_setup_particles);
  initTimer(timer_setup_particles);
  startTimer(timer_setup_particles);
#endif
  SetSources(field, source, q, Ns, dof, L, seed);
#ifndef DISABLE_TIMING
  stopTimer(timer_setup_particles);
  printTimer(stderr, "setup_particles", timer_setup_particles);
  freeTimer(&timer_setup_particles);
#endif


#ifndef DISABLE_TIMING
  stopTimer(timer_setup); // this will be restarted in bbfmm
#endif


#ifdef ENABLE_DIRECT
  /* Compute the field directly */
  DirectCalc3D(field,source,q,Nf,Ns,dof,0,L,phi);
#else
  /* Compute the field using BBFMM */
  bbfmm(field, source, q, Nf, Ns, L, n, phi, l);
#endif

  /* Print results */
#ifndef DISABLE_TIMING
  startTimer(timer_output);
#endif
  output(field, Nf, dof, phi);
#ifndef DISABLE_TIMING
  stopTimer(timer_output);
#endif

#if !defined(FIELDPOINTS_EQ_SOURCES)
  free(source);
#endif
  free(field);
  free(q);
  free(phi);

  /* Finalise timers */
#ifndef DISABLE_TIMING
  printTimer(stderr, "setup", timer_setup); freeTimer(&timer_setup);
  printTimer(stderr, "comp", timer_comp); freeTimer(&timer_comp);
  printTimer(stderr, "upward", timer_upward); freeTimer(&timer_upward);
  printTimer(stderr, "interact", timer_interact); freeTimer(&timer_interact);
  printTimer(stderr, "downward", timer_downward); freeTimer(&timer_downward);
  printTimer(stderr, "output", timer_output); freeTimer(&timer_output);
#endif

  /* Finalise running timer */
#ifndef DISABLE_TIMING
  stopTimer(timer_run);  
  printTimer(stderr, "run", timer_run);
  freeTimer(&timer_run);
#endif

  /* Finalise main timer */
#ifndef DISABLE_TIMING
  stopTimer(timer_main);
  printTimer(stderr, "main", timer_main);
  freeTimer(&timer_main);
#endif

  MESG("Done.\n");
  return 0;
}
