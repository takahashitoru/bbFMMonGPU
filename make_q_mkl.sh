#!/bin/bash

CPU_CLOCK_GHZ=3.324837 # q.nuem.nagoya-u.ac.jp
CC=icc
PLAT=_INTEL120

LDFLAGS="-lm"

OPTS="-std=c99 -O3 -xSSE4.2"
#OPTS="-std=c99 -O3 -xSSE2" # for valgrind
OPTS+=" -mkl=sequential" # enable MKL; file:///opt/intel/composer_xe_2011_sp1.8.273/Documentation/en_US/mkl/mkl_userguide/index.htm
OPTS+=" -vec-report2"
OPTS+=" -Wall"
#OPTS+=" -D_DEBUG -DMYDEBUG -g -check-uninit"

OPTS+=" -openmp"
OPTS+=" -openmp-report2"

#OMP="-lintlc -liomp5 -lpthread"
OMP="-liomp5 -lpthread"

#INCLUDES=" -I/home/ttaka/memcpy_DV"
#LIBS=" -L/home/ttaka/memcpy_DV -lmemcpy_DV${PLAT}"

make -f Makefile CC="${CC}" LDFLAGS="${LDFLAGS}" OPTS="${OPTS}" CPU_CLOCK_GHZ="${CPU_CLOCK_GHZ}" OMP="${OMP}" INCLUDES="${INCLUDES}" LIBS="${LIBS}" $*
