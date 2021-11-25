#!/bin/bash

if [ $# -ne 4 ] ; then
    echo "usage ./bench4b.sh (dir.out BINARY) (tail for dir) (fmm.out BINARY) (tail for fmm)"
    exit
fi

tools=../../tools
if [ ! -f ${tools}/comperr4.c ] ; then
    echo "${tools}/comperr4.c does not exist. Exit."
    exit
fi

direxe=$1 
if [ $2 = "NULL" ] ; then
    dirtail=
else
    dirtail=$2
fi
fmmexe=$3
fmmtail=$4

tmp=${fmmexe%.out}; fmmhead=${tmp#*/}
tmp=${direxe%.out}; dirhead=${tmp#*/}

fmm_precision=$(echo ${fmmhead}  | cut -c1) # first character
dir_precision=$(echo ${dirhead}  | cut -c1) # first character
if [ "${fmm_precision}" != "${dir_precision}" ] ; then
    echo "Precisions differ. Exit."
    exit
else 
    precision=${fmm_precision}
    if [ "${precision}" = "s" ] ; then
	gcc -std=c99 -DSINGLE -o ${precision}comperr4 ${tools}/comperr4.c -lm
    elif [ "${precision}" = "d" ] ; then
	gcc -std=c99          -o ${precision}comperr4 ${tools}/comperr4.c -lm
    else
	echo "Wrong precision. Exit"
	exit
    fi
fi

bench=${dirhead}${dirtail}-${fmmhead}${fmmtail}.bench

echo "# N T_dir T_fmm Error Gflop/s" > $bench

#for N in 100 1000 10000 100000 1000000 10000000
for N in 100 1000 10000 100000 1000000 10000000 100000000
  do
  dirbase=${dirhead}N${N}
  fmmbase=${fmmhead}N${N}
  if [ -f ${dirbase}.err${dirtail} ] ; then
      dirT=$(grep printTimer ${dirbase}.err${dirtail} | grep main | cut -d'=' -f2)
  else
      dirT=-1
  fi 
  if [ -f ${fmmbase}.err${fmmtail} ] ; then
      fmmT=$(grep printTimer ${fmmbase}.err${fmmtail} | grep main | cut -d'=' -f2)
      perf=$(grep -e "calc_performance: kernel " ${fmmbase}.err${fmmtail} | cut -d'=' -f2 | cut -d'[' -f1)
  else 
      fmmT=-1
      perf=-1
  fi
  if [ -f ${dirbase}.out${dirtail} ] && [ -f ${fmmbase}.out${fmmtail} ] ; then
      error=$(./${precision}comperr4 ${N} ${dirbase}.out${dirtail} ${fmmbase}.out${fmmtail} | awk '{print $2}')
  else
      error=-1
  fi
  echo ${N} ${dirT} ${fmmT} ${error} ${perf} >> $bench
done
echo "$0: Created ${bench}"
cat ${bench}
exit
