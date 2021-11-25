#!/bin/bash

if [ $# -lt 2 ] ; then
    echo "usage: ./readtime.sh (header of .err file ) (suffix of .err file)"
    echo "usage: e.g. ./readtime.sh scuda38AD4 -n4-omp8"
    exit 1
fi

###########################################################
# Define N and mofifier for cmidrule
###########################################################
NS="1000 10000 100000 1000000 10000000"; colminus=1  # 10^3--10^7
###########################################################

if [ ! -f ./round ] ; then
    gcc -DENABLE_F_FORMAT -o round ../round.c
fi

if [ ! -f ./eround ] ; then
    gcc -o eround ../round.c
fi

#if [ ! -f ./percentage ] ; then
#    gcc -o percentage ../percentage.c
    gcc -DDIGIT_ONE -o percentage ../percentage.c
#fi

#if [ ! -f ./subsub ] ; then
    gcc -o subsub ../subsub.c
#fi

function filt()
{
    sed s/_/\\\\_/g
}


function ROUND3()
{
    ./round $1 3
}

function ROUND2()
{
    ./round $1 2
}

function ROUND1()
{
    ./round $1 1
}

function EROUND3()
{
    ./eround $1 3
}

function EROUND2()
{
    ./eround $1 2
}

function EROUND1()
{
    ./eround $1 1
}

function TIME()
{
    local base=$1
    local tail=$2
    local tag=$3
    local word=$4
    local skip=$5
    
    echo -en "${tag}    \t" | filt
    for z in ${NS}
    do
        # Time for target
	f=$(egrep "Timer:|\# m2l_host" ${base}N${z}.err${tail} | grep -e "${word}" | cut -d"=" -f2)
	rf=$(EROUND2 $f)
        # Time for reference (main or run)
	ftotal=$(grep Timer ${base}N${z}.err${tail} | grep -e ${reference} | cut -d"=" -f2)
	percentage=$(./percentage ${f} ${ftotal})
	echo -en "& ${rf} \\PC{${percentage}}"
    done
    echo "\\\\%${skip}"
}

function TIME2()
{
    local base=$1
    local tail=$2
    local tag=$3
    local word=$4
    local word2=$5
    local skip=$6
    
    echo -en "${tag}    \t" | filt
    for z in ${NS}
    do
        # Time for target
	f=$(egrep "Timer:|\# m2l_host" ${base}N${z}.err${tail} | grep -e "${word}" | grep -e "${word2}" | cut -d"=" -f2)
	rf=$(EROUND2 $f)
        # Time for reference (main or run)
	ftotal=$(grep Timer ${base}N${z}.err${tail} | grep -e ${reference} | cut -d"=" -f2)
	percentage=$(./percentage ${f} ${ftotal})
	echo -en "& ${rf} \\PC{${percentage}}"
    done
    echo "\\\\%${skip}"
}

function TIME3()
{
    local base=$1
    local tail=$2
    local tag=$3
    local word=$4
    local skip=$5
    
    echo -en "${tag}    \t" | filt
    for z in ${NS}
    do
        # Time for target
	f=$(egrep "Timer:|\# m2l_host" ${base}N${z}.err${tail} | grep -e "${word}" | cut -d"=" -f2)
	rf=$(EROUND2 $f)
        # Time for reference (main-setup-output)
	fmain=$(grep Timer ${base}N${z}.err${tail} | grep -e " main " | cut -d"=" -f2)
	fsetup=$(grep Timer ${base}N${z}.err${tail} | grep -e " setup " | cut -d"=" -f2)
	foutput=$(grep Timer ${base}N${z}.err${tail} | grep -e " output " | cut -d"=" -f2)
	fcuda=$(grep Timer ${base}N${z}.err${tail} | grep -e " cuda " | cut -d"=" -f2)
	if [ ${#fcuda} -eq 0 ] ; then
	    fcuda="0.0e+0"
	fi
	freference=$(./subsub ${fmain} ${fsetup} ${foutput} ${fcuda})
	if [ ${tag} = "running&&" -o ${tag} = "running&&&" ] ; then
	    percentage=$(./percentage ${freference} ${freference})
	    f=${freference}
	    rf=$(EROUND2 ${f})
	else
	    percentage=$(./percentage ${f} ${freference})
	fi
	#echo -en "& ${rf} \\PC{${percentage}}"
	#echo -en "& \$${rf}\$ \\PC{${percentage}}"
	echo -n "& \texttt{${rf}} \\PC{${percentage}}"
    done
    echo "\\\\%${skip}"
}


function TIME4()
{
    local base=$1
    local tail=$2
    local tag=$3
    local word=$4
    local word2=$5
    local skip=$6
    
    echo -en "${tag}    \t" | filt
    for z in ${NS}
    do
        # Time for target
	f=$(egrep "Timer:|\# m2l_host" ${base}N${z}.err${tail} | grep -e "${word}" | grep -e "${word2}" | cut -d"=" -f2)
	rf=$(EROUND2 $f)
        # Time for reference (main-setup-output)
	fmain=$(grep Timer ${base}N${z}.err${tail} | grep -e " main " | cut -d"=" -f2)
	fsetup=$(grep Timer ${base}N${z}.err${tail} | grep -e " setup " | cut -d"=" -f2)
	foutput=$(grep Timer ${base}N${z}.err${tail} | grep -e " output " | cut -d"=" -f2)
	fcuda=$(grep Timer ${base}N${z}.err${tail} | grep -e " cuda " | cut -d"=" -f2)
	if [ ${#fcuda} -eq 0 ] ; then
	    fcuda="0.0e+0"
	fi
	freference=$(./subsub ${fmain} ${fsetup} ${foutput} ${fcuda})
	if [ ${tag} = "running&&" -o ${tag} = "running&&&" ] ; then
	    percentage=$(./percentage ${freference} ${freference})
	    f=${freference}
	    rf=$(EROUND2 ${f})
	else
	    percentage=$(./percentage ${f} ${freference})
	fi
	#echo -en "& ${rf} \\PC{${percentage}}"
	#echo -en "& \$${rf}\$ \\PC{${percentage}}"
	echo -n "& \texttt{${rf}} \\PC{${percentage}}"
    done
    echo "\\\\%${skip}"
}


function PERF()
{
    local base=$1
    local tail=$2
    local tag=$3
    local word=$4

    #echo -n {[${tag}]} | filt
    echo -n "%${tag}" | filt # comment outed
    for z in ${NS}
    do
	f=`grep calc_performance ${base}N${z}.err${tail} | grep -e "${word}" | cut -d"=" -f2 | cut -d"[" -f1`
	rf=`ROUND1 $f`
	echo -en "& ${rf}"
    done
    echo "\\\\%"
}

function BAND()
{
    local base=$1
    local tail=$2
    local tag=$3
    local word=$4

    echo -n ${tag} | filt
    for z in ${NS}
    do
	f=`grep estimate_performacne_bounded_by_bandwidth ${base}N${z}.err${tail} | grep -e "${word}" | cut -d"=" -f2 | cut -d"[" -f1`
	rf=`ROUND1 $f`
	echo -en "\t& ${rf}"
    done
    echo " \\\\%"
}

#echo "1234.56789" | ROUND3
#exit

base=$1
tail=$2

#echo "\hline"
#echo "% $base $tail"

function cmidruleN_M()
{
    local n=$1
    local m=$2
    local skip=$3
    #echo "\\cmidrule{${n}-${m}}"
    echo "\\cmidrule{${n}-${m}}%${skip}"
}
function cpu_cmidrule1()
{
    local skip=$1
    local max=9
    let max="${max}-${colminus}"
    cmidruleN_M 1 ${max} ${skip}
}
function cpu_cmidrule2()
{
    local skip=$1
    local max=9
    let max="${max}-${colminus}"
    cmidruleN_M 2 ${max} ${skip}
}
function cpu_cmidrule3()
{
    local skip=$1
    local max=9
    let max="${max}-${colminus}"
    cmidruleN_M 3 ${max} ${skip}
}
function gpu_cmidrule1()
{
    local skip=$1
    local max=10
    let max="${max}-${colminus}"
    cmidruleN_M 1 ${max} ${skip}
}
function gpu_cmidrule2()
{
    local skip=$1
    local max=10
    let max="${max}-${colminus}"
    cmidruleN_M 2 ${max} ${skip}
}
function gpu_cmidrule3()
{
    local skip=$1
    local max=10
    let max="${max}-${colminus}"
    cmidruleN_M 3 ${max} ${skip}
}
function gpu_cmidrule4()
{
    local skip=$1
    local max=10
    let max="${max}-${colminus}"
    cmidruleN_M 4 ${max} ${skip}
}


function headlines()
{
    local fields=$1
    local sta=$2
    local end=$3
    local skip=$4

    let end="${end}-${colminus}"

    if [ ${colminus} -eq 0 ] ; then
	echo "     ${fields} \\multicolumn{6}{c}{\$N\$}\\\\%${skip}"
	echo "\\cmidrule{${sta}-${end}}%${skip}"
	echo "Item ${fields} \$10^2\$ & \$10^3\$ & \$10^4\$ & \$10^5\$ & \$10^6\$ & \$10^7\$\\\\%${skip}"
    elif [ ${colminus} -eq 1 ] ; then
	echo "     ${fields} \\multicolumn{5}{c}{\$N\$}\\\\%${skip}"
	echo "\\cmidrule{${sta}-${end}}%${skip}"
	echo "Item ${fields}\$10^3\$ & \$10^4\$ & \$10^5\$ & \$10^6\$ & \$10^7\$\\\\%${skip}"
    elif [ ${colminus} -eq 2 ] ; then
	echo "     ${fields}\\multicolumn{4}{c}{\$N\$}\\\\%${skip}"
	echo "\\cmidrule{${sta}-${end}}%${skip}"
	echo "Item ${fields} \$10^4\$ & \$10^5\$ & \$10^6\$ & \$10^7\$\\\\%${skip}"
    else
	echo "Not implemented yet"
	exit
    fi
}

function cpu_head()
{
    local skip=$1
    headlines "&&&" 4 9 ${skip}
}
function gpu_head()
{
    local skip=$1
    headlines "&&&&" 5 10 ${skip}
}



if [ $base = "scuda46L_ij_sw2nr4tpb64ss64-lap" -o $base = "dcuda46L_ij_sw2nr4tpb64ss64-lap" -o $base = "scuda46N_ij_sw2nr4tpb64ss64-lap" -o $base = "dcuda46N_ij_sw2nr4tpb64ss64-lap" ]; then

    #reference=" run " # reference item
    #gpu_head SKIP4 > timing_GPU_head.tabular

#    TIME $base $tail total\&\&\&       " main " SKIP4
#    gpu_cmidrule1                               SKIP4
    TIME3 $base $tail running\&\&\&     " main "
    gpu_cmidrule1
    TIME3 $base $tail \&upward\&\&      " upward "
    gpu_cmidrule2
    TIME3 $base $tail \&interact\&\&    " interact "
    gpu_cmidrule2
    TIME4 $base $tail \&\&m2l_le_sw\&  "sibling"  " timer_m2l_all "
    gpu_cmidrule3
    TIME4 $base $tail \&\&\&kernel     "sibling"  " timer_m2l_kernel "
    gpu_cmidrule4
    TIME4 $base $tail \&\&\&set        "sibling"  " timer_m2l_set "
    gpu_cmidrule4
    TIME4 $base $tail \&\&\&get        "sibling"  " timer_m2l_get "
    gpu_cmidrule4
    TIME4 $base $tail \&\&m2l_gt_sw\&  "ij"       " timer_m2l_all "
    gpu_cmidrule3
    TIME4 $base $tail \&\&\&kernel     "ij"       " timer_m2l_kernel "
    gpu_cmidrule4
    TIME4 $base $tail \&\&\&set        "ij"       " timer_m2l_set "
    gpu_cmidrule4
    TIME4 $base $tail \&\&\&get        "ij"       " timer_m2l_get "
    gpu_cmidrule4
    TIME3 $base $tail \&\&postm2l\&     " time_kernel2 "
    gpu_cmidrule3
    TIME3 $base $tail \&downward\&\&    " downward "
    gpu_cmidrule2
    TIME3 $base $tail \&\&l2l\&         " time_l2l "
    gpu_cmidrule3
    TIME3 $base $tail \&\&nearby\&      " time_direct_all "
    gpu_cmidrule3
    TIME3 $base $tail \&\&\&kernel      " time_direct_kernel "
    gpu_cmidrule4
    TIME3 $base $tail \&\&\&set         " time_direct_set "
    gpu_cmidrule4
    TIME3 $base $tail \&\&\&get         " time_direct_get "
    gpu_cmidrule1
    TIME3 $base $tail \(setup\)\&\&\&       " setup "
    gpu_cmidrule1
    TIME3 $base $tail \(output\)\&\&\&      " output "
    gpu_cmidrule1
    TIME3 $base $tail \(cuda\)\&\&\&        " cuda "

##    PERF $base $tail KERN    " kernel "
##    PERF $base $tail KERN_LE " kernel_low "
##    PERF $base $tail KERN_GT " kernel_high "

elif [ $base = "scuda46J_ij_sw2nr4tpb64ss64-lap" -o $base = "dcuda46J_ij_sw2nr4tpb64ss64-lap" -o $base = "scuda46J_ij_sw2nr4tpb128ss64-lap" -o $base = "dcuda46J_ij_sw2nr4tpb128ss64-lap" -o $base = "scuda46J_ij_sw2nr4tpb128ss128-lap" -o $base = "dcuda46J_ij_sw2nr4tpb128ss128-lap" -o $base = "scuda46J_ij_sw2nr4tpb192ss192-lap" -o $base = "dcuda46J_ij_sw2nr4tpb192ss192-lap" -o $base = "scuda46J_ij_sw2nr4tpb256ss256-lap" -o $base = "dcuda46J_ij_sw2nr4tpb256ss256-lap"  ]; then

    reference=" run " # reference item

    gpu_head SKIP4 > timing_GPU_head.tabular

    TIME $base $tail total\&\&\&       " main " SKIP4
    gpu_cmidrule1                               SKIP4
    TIME $base $tail running\&\&\&     " run "
    gpu_cmidrule1
    TIME $base $tail \&cuda\&\&        " cuda " SKIP4
    gpu_cmidrule2                               SKIP4
    TIME $base $tail \&setup\&\&       " setup " SKIP4
    gpu_cmidrule2                               SKIP4
    TIME $base $tail \&upward\&\&      " upward " SKIP4
    gpu_cmidrule2                                SKIP4
    TIME $base $tail \&interact\&\&    " interact "
    gpu_cmidrule2
    TIME2 $base $tail \&\&m2l_le_sw\&  "sibling"  " timer_m2l_all "
    gpu_cmidrule3
    TIME2 $base $tail \&\&\&kernel     "sibling"  " timer_m2l_kernel "
    gpu_cmidrule4
    TIME2 $base $tail \&\&\&set        "sibling"  " timer_m2l_set "
    gpu_cmidrule4
    TIME2 $base $tail \&\&\&get        "sibling"  " timer_m2l_get "
    gpu_cmidrule4
    TIME2 $base $tail \&\&m2l_gt_sw\&  "ij"       " timer_m2l_all "
    gpu_cmidrule3
    TIME2 $base $tail \&\&\&kernel     "ij"       " timer_m2l_kernel "
    gpu_cmidrule4
    TIME2 $base $tail \&\&\&set        "ij"       " timer_m2l_set "
    gpu_cmidrule4
    TIME2 $base $tail \&\&\&get        "ij"       " timer_m2l_get "
    gpu_cmidrule4
    TIME $base $tail \&\&postm2l\&     " time_kernel2 " SKIP4
    gpu_cmidrule3                                      SKIP4
    TIME $base $tail \&downward\&\&    " downward " SKIP4
    gpu_cmidrule2                                      SKIP4
    TIME $base $tail \&\&l2l\&         " time_l2l " SKIP4
    gpu_cmidrule3                                  SKIP4
    TIME $base $tail \&\&nearby\&      " time_direct_all " SKIP4
    gpu_cmidrule3                                         SKIP4
##    TIME $base $tail \&\&\&setup       " time_direct_setup " SKIP4
##    gpu_cmidrule4                                           SKIP4
    TIME $base $tail \&\&\&kernel      " time_direct_kernel " SKIP4
    gpu_cmidrule4                                            SKIP4
    TIME $base $tail \&\&\&set         " time_direct_set " SKIP4
    gpu_cmidrule4                                         SKIP4
    TIME $base $tail \&\&\&get         " time_direct_get " SKIP4
    gpu_cmidrule2                                         SKIP4
    TIME $base $tail \&output\&\&      " output " SKIP4

##    PERF $base $tail KERN    " kernel "
##    PERF $base $tail KERN_LE " kernel_low "
##    PERF $base $tail KERN_GT " kernel_high "


elif [ $base = "scuda46J_si_tpb128-lap" -o $base = "dcuda46J_si_tpb128-lap" ]; then

    reference=" run " # reference item

    gpu_head SKIP2SKIP3 > timing_GPU_head.tabular

    TIME $base $tail total\&\&\&       " main " SKIP1SKIP2SKIP3
    gpu_cmidrule1                               SKIP1SKIP2SKIP3
#    TIME $base $tail running\&\&\&     " run "
    TIME $base $tail run\&\&\&     " run "
    gpu_cmidrule1
    TIME $base $tail \&cuda\&\&        " cuda " SKIP1SKIP2SKIP3
    gpu_cmidrule2                               SKIP1SKIP2SKIP3
    TIME $base $tail \&setup\&\&       " setup " SKIP2SKIP3
    gpu_cmidrule2                                SKIP2SKIP3
#    TIME $base $tail \&upward\&\&      " upward " SKIP2SKIP3
    TIME $base $tail \&up\&\&      " upward " SKIP2SKIP3
    gpu_cmidrule2                                 SKIP2SKIP3
    TIME $base $tail \&interact\&\&    " interact "
    gpu_cmidrule2
    TIME $base $tail \&\&m2l\&         " timer_m2l_all "
    gpu_cmidrule3
#    TIME $base $tail \&\&\&kernel      " timer_m2l_kernel "
    TIME $base $tail \&\&\&kern      " timer_m2l_kernel "
    gpu_cmidrule4
    TIME $base $tail \&\&\&set         " timer_m2l_set "
    gpu_cmidrule4
    TIME $base $tail \&\&\&get         " timer_m2l_get " 
    gpu_cmidrule4                                       SKIP2SKIP3
#    TIME $base $tail \&\&postm2l\&     " time_kernel2 " SKIP2SKIP3
    TIME $base $tail \&\&pm2l\&     " time_kernel2 " SKIP2SKIP3
    gpu_cmidrule3                                      SKIP2SKIP3
#    TIME $base $tail \&downward\&\&    " downward " SKIP2SKIP3
    TIME $base $tail \&down\&\&    " downward " SKIP2SKIP3
    gpu_cmidrule2                                  SKIP2SKIP3
    TIME $base $tail \&\&l2l\&         " time_l2l " SKIP2SKIP3
    gpu_cmidrule3                                  SKIP2SKIP3
#    TIME $base $tail \&\&nearby\&      " time_direct_all " SKIP2SKIP3
    TIME $base $tail \&\&direct\&      " time_direct_all " SKIP2SKIP3
    gpu_cmidrule3                                         SKIP2SKIP3
###    TIME $base $tail \&\&\&setup       " time_direct_setup " SKIP2SKIP3
###    gpu_cmidrule4                                           SKIP2SKIP3
#    TIME $base $tail \&\&\&kernel      " time_direct_kernel " SKIP2SKIP3
    TIME $base $tail \&\&\&kern      " time_direct_kernel " SKIP2SKIP3
    gpu_cmidrule4                                            SKIP2SKIP3
    TIME $base $tail \&\&\&set         " time_direct_set " SKIP2SKIP3
    gpu_cmidrule4                                         SKIP2SKIP3
    TIME $base $tail \&\&\&get         " time_direct_get " SKIP2SKIP3
    gpu_cmidrule2                                         SKIP2SKIP3
    TIME $base $tail \&output\&\&      " output " SKIP2SKIP3

    PERF $base $tail KERN " kernel "

elif [ $base = "scuda45G_ba-lap" -o $base = "dcuda45G_ba-lap" -o $base = "scuda45G_si-lap" -o $base = "dcuda45G_si-lap" -o $base = "scuda45G_cl-lap" -o $base = "dcuda45G_cl-lap" ]; then

    reference=" run " # reference item

    gpu_head SKIP2SKIP3 > timing_GPU_head.tabular

    TIME $base $tail total\&\&\&       " main " SKIP1SKIP2SKIP3
    gpu_cmidrule1                               SKIP1SKIP2SKIP3
    TIME $base $tail running\&\&\&     " run "
    gpu_cmidrule1
    TIME $base $tail \&cuda\&\&        " cuda " SKIP1SKIP2SKIP3
    gpu_cmidrule2                               SKIP1SKIP2SKIP3
    TIME $base $tail \&setup\&\&       " setup " SKIP2SKIP3
    gpu_cmidrule2                                SKIP2SKIP3
    TIME $base $tail \&upward\&\&      " upward " SKIP2SKIP3
    gpu_cmidrule2                                 SKIP2SKIP3
    TIME $base $tail \&interact\&\&    " interact "
    gpu_cmidrule2
    TIME $base $tail \&\&m2l\&         " timer_m2l_all "
    gpu_cmidrule3
    TIME $base $tail \&\&\&kernel      " timer_m2l_kernel "
    gpu_cmidrule4
    TIME $base $tail \&\&\&set         " timer_m2l_set "
    gpu_cmidrule4
    TIME $base $tail \&\&\&get         " timer_m2l_get " 
    gpu_cmidrule4                                       SKIP2SKIP3
    TIME $base $tail \&\&postm2l\&     " time_kernel2 " SKIP2SKIP3
    gpu_cmidrule3                                      SKIP2SKIP3
    TIME $base $tail \&downward\&\&    " downward " SKIP2SKIP3
    gpu_cmidrule2                                  SKIP2SKIP3
    TIME $base $tail \&\&l2l\&         " time_l2l " SKIP2SKIP3
    gpu_cmidrule3                                  SKIP2SKIP3
    TIME $base $tail \&\&nearby\&      " time_direct_all " SKIP2SKIP3
    gpu_cmidrule3                                         SKIP2SKIP3
    TIME $base $tail \&\&\&setup       " time_direct_setup " SKIP2SKIP3
    gpu_cmidrule4                                           SKIP2SKIP3
    TIME $base $tail \&\&\&kernel      " time_direct_kernel " SKIP2SKIP3
    gpu_cmidrule4                                            SKIP2SKIP3
    TIME $base $tail \&\&\&set         " time_direct_set " SKIP2SKIP3
    gpu_cmidrule4                                         SKIP2SKIP3
    TIME $base $tail \&\&\&get         " time_direct_get " SKIP2SKIP3
    gpu_cmidrule2                                         SKIP2SKIP3
    TIME $base $tail \&output\&\&      " output " SKIP2SKIP3

    PERF $base $tail KERN " kernel "


elif [ $base = "scuda45G_ij_sw2nr4-lap" -o $base = "dcuda45G_ij_sw2nr4-lap" ]; then

    reference=" run " # reference item

    gpu_head SKIP4 > timing_GPU_head.tabular

    TIME $base $tail total\&\&\&       " main " SKIP4
    gpu_cmidrule1                               SKIP4
    TIME $base $tail running\&\&\&     " run "
    gpu_cmidrule1
    TIME $base $tail \&cuda\&\&        " cuda " SKIP4
    gpu_cmidrule2                               SKIP4
    TIME $base $tail \&setup\&\&       " setup " SKIP4
    gpu_cmidrule2                               SKIP4
    TIME $base $tail \&upward\&\&      " upward " SKIP4
    gpu_cmidrule2                                SKIP4
    TIME $base $tail \&interact\&\&    " interact "
    gpu_cmidrule2
    TIME2 $base $tail \&\&m2l_le_sw\&  "sibling"  " timer_m2l_all "
    gpu_cmidrule3
    TIME2 $base $tail \&\&\&kernel     "sibling"  " timer_m2l_kernel "
    gpu_cmidrule4
    TIME2 $base $tail \&\&\&set        "sibling"  " timer_m2l_set "
    gpu_cmidrule4
    TIME2 $base $tail \&\&\&get        "sibling"  " timer_m2l_get "
    gpu_cmidrule4
    TIME2 $base $tail \&\&m2l_gt_sw\&  "ij"       " timer_m2l_all "
    gpu_cmidrule3
    TIME2 $base $tail \&\&\&kernel     "ij"       " timer_m2l_kernel "
    gpu_cmidrule4
    TIME2 $base $tail \&\&\&set        "ij"       " timer_m2l_set "
    gpu_cmidrule4
    TIME2 $base $tail \&\&\&get        "ij"       " timer_m2l_get "
    gpu_cmidrule4
    TIME $base $tail \&\&postm2l\&     " time_kernel2 " SKIP4
    gpu_cmidrule3                                      SKIP4
    TIME $base $tail \&downward\&\&    " downward " SKIP4
    gpu_cmidrule2                                      SKIP4
    TIME $base $tail \&\&l2l\&         " time_l2l " SKIP4
    gpu_cmidrule3                                  SKIP4
    TIME $base $tail \&\&nearby\&      " time_direct_all " SKIP4
    gpu_cmidrule3                                         SKIP4
    TIME $base $tail \&\&\&setup       " time_direct_setup " SKIP4
    gpu_cmidrule4                                           SKIP4
    TIME $base $tail \&\&\&kernel      " time_direct_kernel " SKIP4
    gpu_cmidrule4                                            SKIP4
    TIME $base $tail \&\&\&set         " time_direct_set " SKIP4
    gpu_cmidrule4                                         SKIP4
    TIME $base $tail \&\&\&get         " time_direct_get " SKIP4
    gpu_cmidrule2                                         SKIP4
    TIME $base $tail \&output\&\&      " output " SKIP4

    PERF $base $tail KERN    " kernel "
    PERF $base $tail KERN_LE " kernel_low "
    PERF $base $tail KERN_GT " kernel_high "


#110307elif [ $base = "scuda39A" -o $base = "scuda39A-lap"  -o $base = "scuda39B-lap" -o $base = "scuda39C-lap" -o $base = "scuda41L_si-lap" -o $base = "scuda41L_cl-lap" -o ]; then
#110307 
#110307    reference=" run " # reference item
#110307
#110307    gpu_head SKIP2SKIP3 > timing_GPU_head.tabular
#110307
#110307    TIME $base $tail total\&\&\&       " main " SKIP1SKIP2SKIP3
#110307    gpu_cmidrule1                               SKIP1SKIP2SKIP3
#110307    TIME $base $tail running\&\&\&       " run "
#110307    gpu_cmidrule1
#110307    TIME $base $tail \&cuda\&\&        " cuda " SKIP1SKIP2SKIP3
#110307    gpu_cmidrule2                              SKIP1SKIP2SKIP3
#110307    TIME $base $tail \&setup\&\&       " setup " SKIP2SKIP3
#110307    gpu_cmidrule2                               SKIP2SKIP3
#110307    TIME $base $tail \&upward\&\&      " upward " SKIP2SKIP3
#110307    gpu_cmidrule2                                SKIP2SKIP3
#110307    TIME $base $tail \&interact\&\&    " interact "
#110307    gpu_cmidrule2
#110307    TIME $base $tail \&\&setup\&       " time_conv "
#110307    gpu_cmidrule3
#110307    TIME $base $tail \&\&m2l\&         " time_kernel "
#110307    gpu_cmidrule3
#110307    TIME $base $tail \&\&set\&         " time_set "
#110307    gpu_cmidrule3
#110307    TIME $base $tail \&\&get\&         " time_get " 
#110307    gpu_cmidrule3                                  SKIP2SKIP3
#110307    TIME $base $tail \&\&postm2l\&     " time_kernel2 " SKIP2SKIP3
#110307    gpu_cmidrule3                                      SKIP2SKIP3
#110307    TIME $base $tail \&downward\&\&    " downward " SKIP2SKIP3
#110307    gpu_cmidrule2                                  SKIP2SKIP3
#110307    TIME $base $tail \&\&l2l\&         " time_l2l " SKIP2SKIP3
#110307    gpu_cmidrule3                                  SKIP2SKIP3
#110307    TIME $base $tail \&\&nearby\&      " time_direct_all " SKIP2SKIP3
#110307    gpu_cmidrule3                                         SKIP2SKIP3
#110307    TIME $base $tail \&\&\&setup       " time_direct_setup " SKIP2SKIP3
#110307    gpu_cmidrule4                                           SKIP2SKIP3
#110307    TIME $base $tail \&\&\&kernel      " time_direct_kernel " SKIP2SKIP3
#110307    gpu_cmidrule4                                            SKIP2SKIP3
#110307    TIME $base $tail \&\&\&set         " time_direct_set " SKIP2SKIP3
#110307    gpu_cmidrule4                                         SKIP2SKIP3
#110307    TIME $base $tail \&\&\&get         " time_direct_get " SKIP2SKIP3
#110307    gpu_cmidrule2                                         SKIP2SKIP3
#110307    TIME $base $tail \&output\&\&      " output " SKIP2SKIP3
#110307
#110307    PERF $base $tail KERN " kernel "
#110307
#110307
#110307elif [ $base = "scuda41L2_2nr1-lap" -o $base = "scuda41L2_2nr2-lap" -o $base = "scuda41L2_2nr4-lap" -o $base = "scuda41L2_2nr8-lap" -o $base = "scuda41L2_2nr16-lap" -o $base = "scuda41L2_2nr32-lap" -o $base = "scuda41L2nr4-lap" -o $base = "scuda41L2nr8-lap" -o $base = "scuda41L2nr16-lap" -o $base = "scuda41L2nr32-lap" ]; then
#110307
#110307    reference=" run " # reference item
#110307
#110307    gpu_head SKIP4 > timing_GPU_head.tabular
#110307
#110307    TIME $base $tail total\&\&\&       " main " SKIP4
#110307    gpu_cmidrule1                               SKIP4
#110307    TIME $base $tail running\&\&\&     " run "
#110307    gpu_cmidrule1
#110307    TIME $base $tail \&cuda\&\&        " cuda " SKIP4
#110307    gpu_cmidrule2                               SKIP4
#110307    TIME $base $tail \&setup\&\&       " setup " SKIP4
#110307    gpu_cmidrule2                               SKIP4
#110307    TIME $base $tail \&upward\&\&      " upward " SKIP4
#110307    gpu_cmidrule2                                SKIP4
#110307    TIME $base $tail \&interact\&\&    " interact "
#110307    gpu_cmidrule2
#110307    TIME $base $tail \&\&setup\&       " time_conv "
#110307    gpu_cmidrule3
#110307    TIME $base $tail \&\&m2l\&         " time_kernel "
#110307    gpu_cmidrule3
#110307    TIME $base $tail \&\&\&le_sw       " time_kernel_low "
#110307    gpu_cmidrule4
#110307    TIME $base $tail \&\&\&gt_sw       " time_kernel_high "
#110307    gpu_cmidrule4
#110307    TIME $base $tail \&\&set\&         " time_set "
#110307    gpu_cmidrule3
#110307    TIME $base $tail \&\&get\&         " time_get "
#110307    gpu_cmidrule3                                  SKIP4
#110307    TIME $base $tail \&\&postm2l\&     " time_kernel2 " SKIP4
#110307    gpu_cmidrule3                                      SKIP4
#110307    TIME $base $tail \&downward\&\&    " downward " SKIP4
#110307    gpu_cmidrule2                                      SKIP4
#110307    TIME $base $tail \&\&l2l\&         " time_l2l " SKIP4
#110307    gpu_cmidrule3                                  SKIP4
#110307    TIME $base $tail \&\&nearby\&      " time_direct_all " SKIP4
#110307    gpu_cmidrule3                                         SKIP4
#110307    TIME $base $tail \&\&\&setup       " time_direct_setup " SKIP4
#110307    gpu_cmidrule4                                           SKIP4
#110307    TIME $base $tail \&\&\&kernel      " time_direct_kernel " SKIP4
#110307    gpu_cmidrule4                                            SKIP4
#110307    TIME $base $tail \&\&\&set         " time_direct_set " SKIP4
#110307    gpu_cmidrule4                                         SKIP4
#110307    TIME $base $tail \&\&\&get         " time_direct_get " SKIP4
#110307    gpu_cmidrule2                                         SKIP4
#110307    TIME $base $tail \&output\&\&      " output " SKIP4
#110307
#110307    PERF $base $tail KERN    " kernel "
#110307    PERF $base $tail KERN_LE " kernel_low "
#110307    PERF $base $tail KERN_GT " kernel_high "


elif [ $base = "sfmm7A-lap" -o $base = "dfmm7A-lap" -o $base = "sfmm8A-lap" -o $base = "dfmm8A-lap" ] ; then

    # reference is "main-setup-output"

    #cpu_head > timing_CPU_head.tabular

    TIME3 $base $tail running\&\&   " main "
    cpu_cmidrule1
    TIME3 $base $tail \&upward\&      " upward "
    cpu_cmidrule2
    TIME3 $base $tail \&interact\&    " interact "
    cpu_cmidrule2
    TIME3 $base $tail \&\&m2l         " kernel "
    cpu_cmidrule3
    TIME3 $base $tail \&\&postm2l     " kernel2 "
    cpu_cmidrule2
    TIME3 $base $tail \&downward\&    " downward "
    cpu_cmidrule2
    TIME3 $base $tail \&\&l2l         " downward_l2l "
    cpu_cmidrule3
    TIME3 $base $tail \&\&nearby      " downward_nearby "
    cpu_cmidrule1
    cpu_cmidrule1
    TIME3 $base $tail \(setup\)\&\&       " setup "
    cpu_cmidrule1
    TIME3 $base $tail \(output\)\&\&      " output "

    PERF $base $tail KERN " kernel "

elif [ $base = "sfmm3C32" -o $base = "sfmm5B" -o $base = "sfmm3C32-lap" -o $base = "sfmm5B-lap" -o $base = "sfmm6D-lap" -o $base = "dfmm6D-lap" ] ; then

    reference=" run " # reference item

    cpu_head > timing_CPU_head.tabular

    TIME $base $tail running\&\&   " run "
    cpu_cmidrule1
    TIME $base $tail \&setup\&       " setup "
    cpu_cmidrule2
    TIME $base $tail \&upward\&      " upward "
    cpu_cmidrule2
    TIME $base $tail \&interact\&    " interact "
    cpu_cmidrule2
    TIME $base $tail \&\&m2l         " kernel "
    cpu_cmidrule3
    TIME $base $tail \&\&postm2l     " kernel2 "
    cpu_cmidrule2
    TIME $base $tail \&downward\&    " downward "
    cpu_cmidrule2
    TIME $base $tail \&\&l2l         " downward_l2l "
    cpu_cmidrule3
    TIME $base $tail \&\&nearby      " downward_nearby "
    cpu_cmidrule2
    TIME $base $tail \&output\&      " output "

    PERF $base $tail KERN " kernel "

else
    echo readtime-lap.sh: ERROR Unknown headder \"$base\". ADD it. Exit.
    exit 1
fi

