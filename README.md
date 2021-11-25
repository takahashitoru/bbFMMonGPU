# bbFMMonGPU

The goal of this project is to develop a high-performance black-box fast multipole method (bbFMM) that can run very fast on a CUDA-capable GPU. Our prime objective is to offload the multipole-to-local operation (M2L) and the particle-to-particle computation (P2P) efficiently onto GPU. The bbFMM is the primary target, but other FMM variants can work with our high-performance M2L codes.

==================
How to run bbfmm
==================
--------------------------------------------------------------
0. Please read the file COPYING
--------------------------------------------------------------
Everything for the present program comes with absolutely no warranty.

--------------------------------------------------------------
1. Prepare make_xxx.sh and cumake_xxx.sh
--------------------------------------------------------------

o Edit a template for compiling the bbFMM codes. Choose make_gcc.sh
  and cumake_gcc.sh if you use GCC (GNU Compilers) and make_icc.sh and
  cumake_gcc.sh if ICC (Intel Composers).

o In each template, specify the follwing variables:
  CPU_CLOCK_GHZ: Clock rate of your computer in GHz
  LDFLAGS: Path for BLAS and LAPACK libraries.
  OPTS: Give -fopenmp if GCC and -openmp if ICC in order to enable OpenMP.

--------------------------------------------------------------
2. Edit www.sh adn run it to make executable files
--------------------------------------------------------------
o This script makes CPU, GPU, and direct codes. 
o Set your make_xxx.sh and cumake_xxx.sh.
o Run the script as "./www.sh".

--------------------------------------------------------------
3. Benchmarking
--------------------------------------------------------------
Run the DIRECT code, CPU code, and GPU code in order.

o Go to 'bench' directory.

o Edit the following variables in dir-lap.sh and execute "./dir-lap.sh".

  P: precison; s or d
  exe: Direct code made in section 2
  omp: Number of threads to be used for OpenMP
  N: Number of particles

  You can obtain the following files for each N:

  ${P}{exe}N${N}.{err,out}-n${n}-omp${omp}-m${m}

  where .err contains the log and .out contains the field value
  (binary data).

o Edit the following variables in fmm-lap.sh and execute "./fmm-lap.sh".

  P: precison; s or d
  exe: CPU code made in section 2
  omp: Number of threads for OpenMP
  N: Number of particles
  n: Order of Chebyshev interpolation; 4 or 8
  m: Level modifier; any integer

o Edit cuda-lap.sh similarly to fmm-lap.sh and execute "./cuda-lap.sh".

  .bench gives the time and performance (Gflop/s) of M2L operation,
  while .bench_direct gives the time and performance (G interations/s)
  of direct computation.

- ---
Toru Takahashi
Apr 16, 2012


