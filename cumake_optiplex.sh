#!/bin/bash

CPU_CLOCK_GHZ=2.927 #CPU clock rate at GHz

#CC=gcc-4.4 #C/C++ compiler GCC4.4.6 (same as 'gcc')
CC=gcc-4.6 #C/C++ compiler GCC4.6.1

TK=/usr/local/cuda
SDK=/usr/local/cuda/gpucomputingsdk/C

NVCC=${TK}/bin/nvcc

OPTS="-std=c99 -O3 -msse4.2"
#OPTS+=" -Wall -D_DEBUG -DMYDEBUG -g"
OPTS+=" -fopenmp"

OMP="-lgomp -lpthread" # OpenMP

#PLAT=_GCC446_omp #for GCC4.4.6
PLAT=_GCC461_omp #for GCC4.6.1
LIBS="-L/home/toru/lib -llapack${PLAT} -lblas${PLAT} -lgfortran -lm" # LAPACK BLAS math
 

CUOPTS="-DCUDA --ptxas-options=-v -DC2050" # GPU Model does not matter...
#CUOPTS+=" -D_DEBUG -DMYDEBUG -g"
#CUOPTS+=" -maxrregcount 64"

CUDA_ARCH=20
CUOPTS+=" -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} -DCUDA_ARCH=${CUDA_ARCH}"

CUINCLUDES="-I. -I${SDK}/common/inc"
CULIBS="-L${TK}/lib64 -L${SDK}/lib -lcudart -lcutil_x86_64 -lcuda" #cuda3.2.16, 4.0.17
CULDFLAGS=

make -f Makefile.cuda CC="${CC}" NVCC="${NVCC}" OPTS="${OPTS}" LIBS="${LIBS}" CUOPTS="${CUOPTS}" CULDFLAGS="${CULDFLAGS}" CUINCLUDES="${CUINCLUDES}" CULIBS="${CULIBS}" CPU_CLOCK_GHZ="${CPU_CLOCK_GHZ}" OMP="${OMP}" $*
