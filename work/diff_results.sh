#!/bin/bash

if [ $# -lt 2 ] ; then
    echo "$0 (scudaXX-lap) (scudaYY-lap)"
    exit
fi

base1=$1
base2=$2

for n in 4 8
  do
  for N in 100 1000 10000 100000 1000000 10000000
    do
    tail=-n${n}-omp4
    diff ${base1}N${N}.out${tail}  ${base2}N${N}.out${tail}
  done
done

	
