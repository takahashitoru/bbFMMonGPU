#!/bin/bash

CPU_CLOCK_GHZ=3.324837 #CPU clock rate at GHz
CC=icc #C/C++ compiler

LDFLAGS="-L/opt/intel/composerxe-2011/lib/intel64 -lirc -limf -lsvml -L/usr/lib64 -llapack -lblas -lm"

OPTS="-std=c99 -O3 -xSSE4.2"
OPTS+=" -vec-report2"
#OPTS+=" -Wall -D_DEBUG -DMYDEBUG"
OPTS+=" -openmp"
OPTS+=" -openmp-report2"

OMP="-lintlc -liomp5 -lpthread" # OpenMP

make -f Makefile CC="${CC}" LDFLAGS="${LDFLAGS}" OPTS="${OPTS}" CPU_CLOCK_GHZ="${CPU_CLOCK_GHZ}" OMP="${OMP}" $*
