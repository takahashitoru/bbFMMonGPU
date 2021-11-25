#!/bin/bash

CPU_CLOCK_GHZ=3.324837 #CPU clock rate at GHz
CC=gcc #C/C++ compiler

#LDFLAGS="-L/usr/lib64 -llapack -lblas -lm"
LDFLAGS="-L/home/ttaka/lib -llapack_GCC445_omp -lblas_GCC445_omp -lgfortran -lm"

#OPTS="-std=c99 -O3 -msse3"
OPTS="-std=c99 -O3 -msse4.2"
#OPTS+=" -Wall -D_DEBUG -DMYDEBUG"
OPTS+=" -fopenmp"

OMP="-L/usr/lib64 -lgomp -lpthread" # OpenMP

make -f Makefile CC="${CC}" LDFLAGS="${LDFLAGS}" OPTS="${OPTS}" CPU_CLOCK_GHZ="${CPU_CLOCK_GHZ}" OMP="${OMP}" $*
