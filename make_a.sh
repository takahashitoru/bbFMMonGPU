#!/bin/bash



CPU_CLOCK_GHZ=3.10 #CPU clock rate at GHz
#CC=/home/ttaka/gcc-4.6.1/bin/gcc #C/C++ compiler
#CC=gcc #gcc-4.5.1 Default
CC=icc

#LDFLAGS="-L/usr/lib64 -llapack -lblas -lm"
#LDFLAGS="-L/home/ttaka/lib -llapack_GCC461_omp -lblas_GCC461_omp -L/home/ttaka/gcc-4.6.1/lib64 -lgfortran -lm" # Does not work
#LDFLAGS="-L/home/ttaka/lib -L/usr/lib64 -llapack -lblas -lgfortran -lm" # export LD_LIBRARY_PATH=/home/ttaka/gcc-4.6.1/lib64:$LD_LIBRARY_PATH
#LDFLAGS="-L/home/ttaka/lib -llapack_GCC451_omp -lblas_GCC451_omp -L/usr/lib64 /usr/lib64/gcc/i686-pc-mingw32/4.5.0 -lgfortran  -lm"
#LDFLAGS="-L/opt/intel/composerxe/lib/intel64 -lirc -limf -lsvml -lifcoremt -L/home/ttaka/lib -llapack_INTEL121_omp -lblas_INTEL121_omp -lm" # bad?
#LDFLAGS="-L/opt/intel/composerxe/lib/intel64 -lirc -limf -lsvml -L/usr/lib64 -llapack -lblas -lm" # same as make_q.sh

#LDFLAGS="-L/opt/intel/composer_xe_2011_sp1.7.256/compiler/lib/intel64 -L/usr/lib64 -llapack -lblas -lm" # -lirc -limf -lsvml can be omitted
#LDFLAGS="-L/usr/lib64 -llapack -lblas -lm" # -lirc -limf -lsvml can be removed

#LDFLAGS=" -L/home/ttaka/lib -llapack_INTEL121_omp -lblas_INTEL121_omp /opt/intel/composer_xe_2011_sp1.8.273/compiler/lib/intel64/libifcore.a -lm" # SORGLQ stops the run
#LDFLAGS=" -L/home/ttaka/lib -llapack_INTEL121 -lblas_INTEL121 /opt/intel/composer_xe_2011_sp1.8.273/compiler/lib/intel64/libifcore.a -lm" # SORGLQ stops the run
#LDFLAGS=" /home/ttaka/lapack-3.2.2/liblapack_INTEL121_omp.a /home/ttaka/lapack-3.2.2/libblas_INTEL121_omp.a /opt/intel/composer_xe_2011_sp1.8.273/compiler/lib/intel64/libifcore.a -lm" # NG
#LDFLAGS=" /home/ttaka/lapack-3.2.2/liblapack_INTEL121.a /home/ttaka/lapack-3.2.2/libblas_INTEL121.a /opt/intel/composer_xe_2011_sp1.8.273/compiler/lib/intel64/libifcore.a -lm"

#LDFLAGS=" /opt/intel/composer_xe_2011_sp1.8.273/mkl/lib/intel64/libmkl_intel_ilp64.a"
#LDFLAGS+=" /opt/intel/composer_xe_2011_sp1.8.273/mkl/lib/intel64/libmkl_sequential.a"
#LDFLAGS+=" /opt/intel/composer_xe_2011_sp1.8.273/mkl/lib/intel64/libmkl_core.a"
#LDFLAGS+=" /opt/intel/composer_xe_2011_sp1.8.273/compiler/lib/intel64/libifcore.a -lm"
LDFLAGS+=" -lm"

#OPTS="-std=c99 -O3 -xAVX" # This seems to give STRANGE results
#OPTS="-std=c99 -O3 -xSSE4.2"
OPTS="-std=c99 -O3 -mkl=sequential"
#OPTS="-std=c99 -O0"
OPTS+=" -vec-report2"
#OPTS+=" -Wall -D_DEBUG -DMYDEBUG"
OPTS+=" -openmp"

#OMP="-L/usr/lib64 -lgomp -lpthread" # OpenMP
#OMP="-lintlc -liomp5 -lpthread"
#OMP="-L/opt/intel/composer_xe_2011_sp1.7.256/compiler/lib/intel64 -liomp5 -lpthread" # OpenMP
OMP="-liomp5 -lpthread" # OpenMP

make -f Makefile CC="${CC}" LDFLAGS="${LDFLAGS}" OPTS="${OPTS}" CPU_CLOCK_GHZ="${CPU_CLOCK_GHZ}" OMP="${OMP}" $*
