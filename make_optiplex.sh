#!/bin/bash

CPU_CLOCK_GHZ=2.927 #CPU clock rate at GHz

#CC=gcc-4.4 #C/C++ compiler 4.4.6 (same as 'gcc')
CC=gcc-4.6 #C/C++ compiler 4.6.1

#LDFLAGS="-L/usr/lib64 -llapack -lblas -lm"

#LDFLAGS="-L/usr/lib/atlas-base -llapack -lblas -latlas -lgfortran -lm" # this can compile the code, but does not work with OpenMP...  perphaps, multithreaded version must be used.

#PLAT=_GCC446_omp # for GCC4.4.6
PLAT=_GCC461_omp # for GCC4.6.1
LDFLAGS="-L/home/toru/lib -llapack${PLAT} -lblas${PLAT} -lgfortran -lm"

OPTS="-std=c99 -O3 -msse4.2"
#OPTS+=" -Wall -D_DEBUG -DMYDEBUG"
OPTS+=" -fopenmp"

#OMP="-L/usr/lib64 -lgomp -lpthread" # OpenMP
OMP="-lgomp -lpthread" # OpenMP; this is from /usr/lib/gcc/x86_64-linux-gnu/4.4.6/

make -f Makefile CC="${CC}" LDFLAGS="${LDFLAGS}" OPTS="${OPTS}" CPU_CLOCK_GHZ="${CPU_CLOCK_GHZ}" OMP="${OMP}" $*
