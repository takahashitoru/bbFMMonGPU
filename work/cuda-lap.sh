#!/bin/bash

ulimit -s unlimited

##################################################################
# Precision of codes (s: Single-precision, d: Double-precision)
##################################################################
#for P in s
#for P in d
for P in s d
do

######################
# Reference codes
######################
direxe=../${P}dir46-lap.out; dirtail=-omp12 #120208 111203

fmmexe_n4=../${P}fmm8A-lap.out #120208 111108
fmmexe_n8=../${P}fmm8A-lap.out #120208 111108

######################
# CUDA codes
######################
sw=2; nr=4; tpb=64; ss=64; exe=../${P}cuda46Q_ij_sw${sw}nr${nr}tpb${tpb}ss${tpb}-lap.out #120208 111213 ij

ntrial=1
#ntrial=10

if [ ! -f ${exe} ] ; then
    echo "$0: ${exe} does not exist. Exit."
    exit
else
    echo "$0: Run ${exe}"
fi
tmp=${exe%.out}; head=${tmp#*/}

L=1
##l="-1"

if [ "${P}" = "s" ] ; then
    Nlist="100 1000 10000 100000"
    #Nlist="100 1000 10000 100000 1000000 10000000 100000000"
    #Nlist="1048576"
else
    Nlist="100 1000 10000 100000"
    #Nlist="100 1000 10000 100000 1000000 10000000"
    #Nlist="1048576"
fi

#for omp in 1
for omp in 12
do

    if [ ${omp} = 0 ] ; then
	echo "$0: Non-parallel version is not supposed"
	exit
    fi

    for n in 4
    #for n in 4 8
    do

	if [ ${n} -eq 4 ] ; then
	    #dlist="+1 -0 -1"
	    dlist="-0"
	elif [ ${n} -eq 8 ] ; then
	    #dlist="-0 -1 -2"
	    dlist="-1"
	fi

	for d in ${dlist}
	do
	
	    tail=-n${n}-omp${omp}-d${d}

	    for (( itrial=0; itrial<ntrial; itrial++ ))
	    do

		if [ ${ntrial} -gt 1 ] ; then
		    echo "$0: itrial=${itrial}..."
		    tailx=${tail}-trial${itrial}
		    storedir=${head}${tail}
		    if [ ! -d ${storedir} ] ; then
			mkdir ${storedir}
		    fi
		else
		    tailx=${tail}
		fi

		for N in ${Nlist}

		do
		    echo "$0: omp=${omp} n=${n} N=${N} d=${d}"		    
		    base=${head}N${N}

		    outfile=${base}.out${tailx}
		    errfile=${base}.err${tailx}

		    (OMP_NUM_THREADS=${omp} $exe -d ${d} ${N} ${L} ${n} > ${outfile}) &> ${errfile}

		    echo "$0: Created ${errfile}"
		done

		./bench4b.sh ${direxe} ${dirtail} ${exe} ${tailx}

		if [ ${n} -eq 4 ] ; then
		    ./bench5b.sh ${fmmexe_n4} ${tail} ${exe} ${tailx} #fmm is averaged
		elif [ ${n} -eq 8 ] ; then
		    ./bench5b.sh ${fmmexe_n8} ${tail} ${exe} ${tailx} #fmm is averaged
		else
		    echo "$0: Unexpected n. Exit"
		    exit
		fi

		./bench_direct.sh ${exe} ${tailx} # this is not averaged later

                # move the created .out and .err files for this trial
		if [ ${ntrial} -gt 1 ] ; then
		    mv ${head}N*.out${tailx} ${storedir}
		    mv ${head}N*.err${tailx} ${storedir}
		    #echo "$0: Moved ${base}.{out,err}${tailx} to ${storedir}"
		fi

	    done #end of loop over trial

	    if [ ${ntrial} -gt 1 ] ; then
		tmptmp=${direxe%.out}; dirhead=${tmptmp#*/}
		if [ "${dirtail}" != "NULL" ] ; then
		    benchhead=${dirhead}${dirtail}-${storedir}
		else
		    benchhead=${dirhead}-${storedir}
		fi
		./bench4b_average.sh ${benchhead}-trial*.bench
		mv ${benchhead}-trial*.bench ${storedir}
		if [ ${n} -eq 4 ] ; then
		    tmptmp=${fmmexe_n4%.out}; fmmexe_n4_head=${tmptmp#*/}
		    ./bench5b_average.sh ${fmmexe_n4_head}${tail}-${head}${tail}-trial*.bench
		    mv ${fmmexe_n4_head}${tail}-${head}${tail}-trial*.bench ${storedir}
		elif [ ${n} -eq 8 ] ; then
		    tmptmp=${fmmexe_n8%.out}; fmmexe_n8_head=${tmptmp#*/}
		    ./bench5b_average.sh ${fmmexe_n8_head}${tail}-${head}${tail}-trial*.bench
		    mv ${fmmexe_n8_head}${tail}-${head}${tail}-trial*.bench ${storedir}
		else
		    echo "$0: Unexpected n. Exit"
		    exit
		fi

		./errfile_average.sh ${head} ${n} ${omp} ${d}

	    fi
	done # end of loop over d
    done # end of loop over n
done # end of loop over omp
done # end of loop over P

echo "$0: done."
exit
