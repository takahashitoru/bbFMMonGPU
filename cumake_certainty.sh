#!/bin/bash

CPU_CLOCK_GHZ=2.666849 # compute-x-x @ Certainty
CC=icc

TK=/usr/local/cuda
SDK=~/NVIDIA_GPU_Computing_SDK_3.2.16/C

NVCC=${TK}/bin/nvcc

OPTS="-O3 -xSSE4.2"
OPTS+=" -vec-report=2"
#OPTS+=" -Wall -D_DEBUG -DMYDEBUG -g -check-uninit"
OPTS+=" -openmp"
OPTS+=" -openmp-report2"

OMP="-lintlc -liomp5 -lpthread"

#PLAT=_INTEL120
#LIBS="-L/share/apps/intel/lib/intel64 -limf -lsvml -llapack -lblas -lm" # either -lifcore nor -lifcoremt is necessary
PLAT=_INTEL120_omp
LIBS="-L/share/apps/intel/lib/intel64 -lifcoremt -lirc -limf -lsvml -L/home/ttaka/lib -llapack${PLAT} -lblas${PLAT} -lm" # either -lifcore nor -lifcoremt is necessary


CUOPTS="-DCUDA --ptxas-options=-v -DC2050"

#CUOPTS+=" -D_DEBUG -DMYDEBUG -g"
#CUOPTS+=" -D_DEBUG -DMYDEBUG -g --device-debug 0"
#CUOPTS+=" -g --device-debug 0"
#CUOPTS+=" -maxrregcount 64"

CUDA_ARCH=20
CUOPTS+=" -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} -DCUDA_ARCH=${CUDA_ARCH}"

CUINCLUDES="-I. -I${SDK}/common/inc"
CULIBS="-L${TK}/lib64 -L${SDK}/lib -lcudart -lcutil_x86_64 -lcuda" #cuda3.2.19
CULDFLAGS=

make -f Makefile.cuda CC="${CC}" NVCC="${NVCC}" OPTS="${OPTS}" LIBS="${LIBS}" CUOPTS="${CUOPTS}" CULDFLAGS="${CULDFLAGS}" CUINCLUDES="${CUINCLUDES}" CULIBS="${CULIBS}" CPU_CLOCK_GHZ="${CPU_CLOCK_GHZ}" OMP="${OMP}" $*
