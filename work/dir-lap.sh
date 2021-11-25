#!/bin/bash

ulimit -s unlimited

##################################################################
# Precision of codes (s: Single-precision, d: Double-precision)
##################################################################
for P in s
#for P in d
#for P in s d
do

######################
# DIRECT code
######################
exe=../${P}dir46-lap.out #120208 111108

tmp=${exe%.out}; head=${tmp#*/}

L=1

#omp=1
#omp=12
#omp=4
omp=6

tail=-omp${omp}

#for N in 100 1000 10000 100000
for N in 100 1000 10000 100000 1000000
#for N in 1048576 #2^20
#for N in 2097152 #2^21
  do
    echo "$0: Processing N=${N}"
    #base=${head}-N${N}
    base=${head}N${N}
    outfile=${base}.out${tail}
    errfile=${base}.err${tail}
    if [ 1 = 1 ] ; then
	(OMP_NUM_THREADS=${omp} ${exe} ${N} ${L} 0 > ${outfile}) >& ${errfile}
    else
	(OMP_NUM_THREADS=${omp} $exe -s 100 ${N} ${L} 0 > ${outfile}) >& ${errfile} #set RANDOM_SEED
    fi
    echo "$0: Created ${base}.err${tail}"
done

done #loop over P

echo "$0: done."
