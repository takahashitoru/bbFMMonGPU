#!/bin/bash

if [ $# -ne 4 ] ; then
    echo "usage: ./bench5b.sh (exeA.out) (tailA) (exeB.out) (tailB)"
    echo "Both outputs (.out*) are supposed to be binary."
    exit
fi

tools=../../tools

exeA=$1 
tailA=$2
exeB=$3
tailB=$4

tmp=${exeA%.out}; headA=${tmp#*/}
tmp=${exeB%.out}; headB=${tmp#*/}

#echo ${exeA} ${exeB} ${headA} ${headB}
#exit

exeA_precision=$(echo ${headA}  | cut -c1) # first character
exeB_precision=$(echo ${headB}  | cut -c1) # first character
if [ "${exeA_precision}" != "${exeB_precision}" ] ; then
    echo "Precisions differ. Exit."
    exit
else 
    precision=${exeA_precision}
    if [ "${precision}" == "s" ] ; then
	gcc -std=c99 -DSINGLE -o ${precision}comperr4 ${tools}/comperr4.c -lm
    elif [ "${precision}" == "d" ] ; then
	gcc -std=c99          -o ${precision}comperr4 ${tools}/comperr4.c -lm
    else
	echo "Wrong precision. Exit"
	exit
    fi
fi


bench=${headA}${tailA}-${headB}${tailB}.bench

echo "% N TimeA TimeB PerfA PerfB Error" > $bench

#for N in 100 1000 10000 100000 1000000 10000000
for N in 100 1000 10000 100000 1000000 10000000 100000000
  do
  baseA=${headA}N${N}
  baseB=${headB}N${N}
  outfileA=${baseA}.out${tailA}
  outfileB=${baseB}.out${tailB}
  errfileA=${baseA}.err${tailA}
  errfileB=${baseB}.err${tailB}
  
  if [ -f ${errfileA} ] ; then
      userA=$(grep printTimer ${errfileA} | grep main | cut -d'=' -f2)
      perfA=$(grep -e "calc_performance: kernel " ${errfileA} | cut -d'=' -f2 | cut -d'[' -f1)
  else
      echo "${errfileA} does not exist."
      userA=-1
      perfA=-1
  fi 
  if [ -f ${errfileB} ] ; then
      userB=$(grep printTimer ${errfileB} | grep main | cut -d'=' -f2)
      perfB=$(grep -e "calc_performance: kernel " ${errfileB} | cut -d'=' -f2 | cut -d'[' -f1)
  else 
      echo "${errfileB} does not exist."
      userB=-1
      perfB=-1
  fi
  if [ -f ${outfileA} ] && [ -f ${outfileB} ] ; then
      error=$(./${precision}comperr4 ${N} ${outfileA} ${outfileB} | awk '{print $2}')
  else
      error=-1
  fi
  echo ${N} ${userA} ${userB} ${perfA} ${perfB} ${error} >> $bench
done
echo "$0: Created ${bench}"
cat ${bench}
exit
