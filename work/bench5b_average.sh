#!/bin/bash

if [ $# -lt 1 ] ; then
    echo "usage: $0 (xxx-trial*.bench)"
    exit
fi

gcc -o e2f ../e2f.c
gcc -o f2e ../f2e.c

firstfile=$1
base=${firstfile%-trial*}
bench=${base}.bench

#echo ${base}

echo "% This file was created by $0 with the following files:" > ${bench}
for file in $*
do
    echo "% $file" >> ${bench}
done
echo "% N  TimeA  TimeB(ave)  PerfA  PerfB(ave)  Error" >> ${bench}

for N in 100 1000 10000 100000 1000000 10000000 100000000
do

    nfile=0
    aveTimeB=0.0
    avePerfB=0.0
    for file in $*
    do

	line=$(grep -e "^${N} " ${file})

	TimeB=$(echo ${line} | awk '{print $3}')
	tmp=$(./e2f ${TimeB})
	aveTimeB=$(echo "${aveTimeB}+${tmp}" | bc -l)
	
	PerfB=$(echo ${line} | awk '{print $5}')
	avePerfB=$(echo "${avePerfB}+${PerfB}" | bc -l)
	
	let nfile=${nfile}+1

    done

    if [ ${nfile} -gt 0 ] ; then
	aveTimeB=$(echo "${aveTimeB}/${nfile}" | bc -l)
	avePerfB=$(echo "${avePerfB}/${nfile}" | bc -l)
	line=$(grep -e "^${N} " ${firstfile})
	TimeA=$(echo ${line} | awk '{print $2}')
	PerfA=$(echo ${line} | awk '{print $4}')
	Error=$(echo ${line} | awk '{print $6}')
    else
	aveTimeB=-1
	avePerfB=-1
	TimeA=-1
	PerfA=-1
	Error=-1
    fi
    
    echo "${N} ${TimeA} $(./f2e ${aveTimeB}) ${PerfA} ${avePerfB} ${Error}" >> ${bench}
    
done

echo "$0: Created ${bench}"
cat ${bench}

exit

