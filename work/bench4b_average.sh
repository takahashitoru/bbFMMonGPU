#!/bin/bash

if [ $# -lt 1 ] ; then
    echo "usage: ./bench4b_average.sh (sdir-xxx-trial*.bench)"
    exit
fi

gcc -o e2f ../e2f.c
gcc -o f2e ../f2e.c

firstfile=$1
base=${firstfile%-trial*}
bench=${base}.bench


echo "% This file was created by average_bench.sh with files:" > ${bench}
for file in $*
do
    echo "% $file" >> ${bench}
done
echo "% N T_dir T_fmm(ave) Error Gflop/s(ave)" >> ${bench}

for N in 100 1000 10000 100000 1000000 10000000 100000000
do
    nfile=0
    aveT_fmm=0.0
    avePerf_fmm=0.0
    for file in $*
    do
	line=$(grep -e "^${N} " ${file})

	T_fmm=$(echo ${line} | awk '{print $3}')
	tmp=$(./e2f ${T_fmm})
	aveT_fmm=$(echo "${aveT_fmm}+${tmp}" | bc -l)

	Perf_fmm=$(echo ${line} | awk '{print $5}')
	avePerf_fmm=$(echo "${avePerf_fmm}+${Perf_fmm}" | bc -l)

	let nfile=${nfile}+1
    done

    if [ ${nfile} -gt 0 ] ; then
	aveT_fmm=$(echo "${aveT_fmm}/${nfile}" | bc -l)
	avePerf_fmm=$(echo "${avePerf_fmm}/${nfile}" | bc -l)
	line=$(grep -e "^${N} " ${firstfile})
	T_dir=$(echo ${line} | awk '{print $2}')
	Error=$(echo ${line} | awk '{print $4}')
    else
	aveT_fmm=-1
	avePerf_fmm=-1
	line=-1
	T_dir=-1
	Error=-1
    fi

    echo "${N} ${T_dir} $(./f2e ${aveT_fmm}) ${Error} ${avePerf_fmm}" >> ${bench}

done

echo "$0: Created ${bench}"
cat ${bench}

exit

