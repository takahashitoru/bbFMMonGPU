#!/bin/bash

if [ $# -ne 2 ] ; then
    echo "usage ./bench_direct.sh (fmm.out BINARY) (tail)"
    exit
fi

tools=../../tools

fmmexe=$1
tail=$2

tmp=${fmmexe%.out}; fmmhead=${tmp#*/}

bench=${fmmhead}${tail}.bench_direct

echo "# (N) (all) (set) (kernel) (get) (G inter/s)" > $bench

#for N in 100 1000 10000 100000 1000000 10000000
for N in 100 1000 10000 100000 1000000 10000000 100000000
  do

  fmmbase=${fmmhead}N${N}

  errfile=$fmmbase.err${tail}
  
  if [ -f ${errfile} ] ; then

      #time_direct_all=$(grep printEventTimer ${errfile} | grep time_direct_all | cut -d'=' -f2)
      #time_direct_set=$(grep printEventTimer ${errfile} | grep time_direct_set | cut -d'=' -f2)
      #time_direct_kernel=$(grep printEventTimer ${errfile} | grep time_direct_kernel | cut -d'=' -f2)
      #time_direct_get=$(grep printEventTimer ${errfile} | grep time_direct_get | cut -d'=' -f2)
      time_direct_all=$(grep time_direct_all ${errfile} | cut -d'=' -f2)
      time_direct_set=$(grep time_direct_set ${errfile} | cut -d'=' -f2)
      time_direct_kernel=$(grep time_direct_kernel ${errfile} | cut -d'=' -f2)
      time_direct_get=$(grep time_direct_get ${errfile} | cut -d'=' -f2)
      direct_perf=$(grep num_pairwise_interactions_per_sec ${errfile} | cut -d'=' -f2 | cut -d'[' -f1)

  else 

      time_direct_all=-1
      time_direct_set=-1
      time_direct_kernel=-1
      time_direct_get=-1
      direct_perf=-1

  fi

  echo ${N} ${time_direct_all} ${time_direct_set} ${time_direct_kernel} ${time_direct_get} ${direct_perf} >> $bench

done
echo "$0: Created ${bench}"
cat ${bench}
exit
