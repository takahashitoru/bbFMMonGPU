#!/bin/bash

CPU_CLOCK_GHZ=3.324837 #CPU clock rate at GHz
CC=icc #C/C++ compiler

TK=/usr/local/cuda4.0.17/cuda
SDK=/usr/local/NVIDIA_CUDA_SDK_4.0.17/C

NVCC=${TK}/bin/nvcc

OPTS="-std=c99 -O3 -xSSE4.2"
OPTS+=" -vec-report=2"
#OPTS+=" -Wall -D_DEBUG -DMYDEBUG -g -check-uninit"
OPTS+=" -openmp"
OPTS+=" -openmp-report2"

OMP="-lintlc -liomp5 -lpthread"

LIBS="-L/opt/intel/composerxe-2011/lib/intel64 -limf -lsvml -L/usr/lib64 -llapack -lblas -lm"


CUOPTS="-DCUDA --ptxas-options=-v -DC2050"
#CUOPTS+=" -D_DEBUG -DMYDEBUG -g"
#CUOPTS+=" -D_DEBUG -DMYDEBUG -g --device-debug 0"
#CUOPTS+=" -maxrregcount 64"

CUDA_ARCH=20
CUOPTS+=" -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} -DCUDA_ARCH=${CUDA_ARCH}"

CUINCLUDES="-I. -I${SDK}/common/inc"
CULIBS="-L${TK}/lib64 -L${SDK}/lib -lcudart -lcutil_x86_64 -lcuda" #cuda3.2.16, 4.0.17
CULDFLAGS=

make -f Makefile.cuda CC="${CC}" NVCC="${NVCC}" OPTS="${OPTS}" LIBS="${LIBS}" CUOPTS="${CUOPTS}" CULDFLAGS="${CULDFLAGS}" CUINCLUDES="${CUINCLUDES}" CULIBS="${CULIBS}" CPU_CLOCK_GHZ="${CPU_CLOCK_GHZ}" OMP="${OMP}" $*
