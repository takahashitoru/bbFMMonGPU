#!/bin/bash

NVCC=/usr/local/cuda4.0.17/cuda/bin/nvcc

SRC=direct; MA=CUDA_VER46; MI=CUDA_VER46Q; OPTS="-DLAPLACIAN -DDIRECT_NUM_THREADS_PER_BLOCK=64 -gencode arch=compute_20,code=sm_20 -DCUDA_ARCH=20"

CUBIN=${SRC}_${MI}.cubin
PTX=${SRC}_${MI}.ptx
DECUDA=${SRC}_${MI}.decuda

if [ 1 == 1 ] ; then
echo -n "Creating ${PTX} ... "
${NVCC} -D${MA} -D${MI} ${OPTS} --ptxas-options=-v --ptx -o ${PTX} ${SRC}.cu
echo "done."
fi

if [ 0 == 1 ] ; then
echo -n "Creating ${CUBIN} ... "
${NVCC} -D${MA} -D${MI} -D${OPTS} --cubin -o ${CUBIN} ${SRC}.cu
echo "done."
echo "Disassembling ${CUBIN} ..."
decuda -p ${CUBIN} > ${DECUDA}
echo "done."
fi

