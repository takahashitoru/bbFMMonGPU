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
# Reference codes
######################
#direxe=../${P}dir46-lap.out; dirtail=-omp12 #120208
#direxe=../${P}dir46-lap.out; dirtail=-omp4 #120208
direxe=../${P}dir46-lap.out; dirtail=-omp6 #120208

######################
# FMM codes
######################
exe=../${P}fmm8A-lap.out #120208 111108
#exe=../${P}fmm9A-lap.out #120214
#exe=../${P}fmm9B-lap.out #120214
#exe=../${P}fmm9C-lap.out #120214
#exe=../${P}fmm9D-lap.out #120214
#exe=../${P}fmm9E-lap.out #120214
#exe=../${P}fmm9F-lap.out #120214
#exe=../${P}fmm9G-lap.out #120214
#exe=../${P}fmm9H-lap.out #120214
#exe=../${P}fmm9I-lap.out #120214
#exe=../${P}fmm9J-lap.out #120214
#exe=../${P}fmm9K-lap.out #120214
#exe=../${P}fmm9L-lap.out #120214
#exe=../${P}fmm9M-lap.out #120214
#exe=../${P}fmm9N-lap.out #120214
#exe=../${P}fmm9O-lap.out #120214
#exe=../${P}fmm9P-lap.out #120214
#exe=../${P}fmm9Q-lap.out #120214
#exe=../${P}fmm9R-lap.out #120214
#exe=../${P}fmm9S-lap.out #120214
#exe=../${P}fmm9T-lap.out #120214
#exe=../${P}fmm9U-lap.out #120214


if [ ! -f ${exe} ] ; then
    echo "$0: ${exe} does not exist. Exit."
    exit
else
    echo "$0: Run ${exe}"
fi

tmp=${exe%.out}; head=${tmp#*/}

######################
# Number to repeat the code
######################
ntrial=1
#ntrial=10

L=1 # Domain size

if [ "${P}" = "s" ] ; then
    #Nlist="100 1000 10000 100000"
    #Nlist="100 1000 10000 100000 1000000 10000000"
    Nlist="100 1000 10000 100000 1000000 10000000 100000000"
else
    Nlist="100 1000 10000 100000"
#    Nlist="100 1000 10000 100000 1000000 10000000"
fi


#for omp in 1
#for omp in 12
#for omp in 4
for omp in 6
do

    if [ ${omp} = 0 ] ; then
	echo "$0: Non-parallel version is not supposed. Exit."
	exit
    fi
    
    #for n in 4 8
    for n in 4
    #for n in 8
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

		    (OMP_NUM_THREADS=${omp} ${exe} -d ${d} ${N} ${L} ${n} > ${outfile}) &> ${errfile}
#		    (OMP_NUM_THREADS=${omp} valgrind ${exe} -d ${d} ${N} ${L} ${n} > ${outfile}) &> ${errfile}

		    echo "$0: Created ${errfile}"
		done

		./bench4b.sh ${direxe} ${dirtail} ${exe} ${tailx}

		# move the created .out and .err files for this trial
		if [ ${ntrial} -gt 1 ] ; then
		    mv ${head}N*.out${tailx} ${storedir}
		    mv ${head}N*.err${tailx} ${storedir}
		    #echo "Moved ${base}.{out,err}${tailx} to ${storedir}"
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
		for N in 100 1000 10000 100000 1000000 10000000 100000000
		do
		    base=${head}N${N}
		    outfile=${base}.out${tail}
		    if [ -f ${storedir}/${outfile}-trial0 ] ; then
			ln -fs ${storedir}/${outfile}-trial0 ${outfile} # link to itrial=0
		    fi
		done

		./errfile_average.sh ${head} ${n} ${omp} ${d}

	    fi

	done # end of loop over d
    done # end of loop over n
done # end of loop over omp
done # end of loop over P

echo "$0: done."
exit

