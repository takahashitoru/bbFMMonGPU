#!/bin/bash

head=$1
n=$2
omp=$3
d=$4

gcc -o average average.c

function average_printEventTimer()
{
    local trials=("$@") #array
    local ntrials=$#
    if [ $ntrials -eq 0 ] ; then
	echo "No files?"
	exit
    fi
    ###############################################
    # Create list of items from the first file
    ###############################################
    #echo "create list of items"
    #echo "There are ${ntrials} files."
    #echo ${trials[@]}
    #echo ${trials[0]}
    #cat ${trials[0]}
    grep printEventTimer ${trials[0]} | cut -d ":" -f2 | cut -d "=" -f1 > tmpfile_items
    #egrep "\# m2l_host" ${trials[0]} | cut -d "#" -f2 | sed s/"m2l_host_"/""/ | sed s/"_blocking: "/"_"/ | cut -d "=" -f1 >> tmpfile_items
    egrep "\# m2l_host" ${trials[0]} | cut -d "#" -f2 | sed s/"m2l_host_"/""/ | sed s/"_blocking"/""/ | sed s/": "/"_"/ | cut -d "=" -f1 >> tmpfile_items
    #cat tmpfile_items
    items=($(cat tmpfile_items | xargs)) #array
    #echo "There are ${#items[*]} items: ${items[@]}"
    if [ ${#items[*]} -eq 0 ] ; then
	return 	# No timer is found
    fi
    ###############################################
    # Create list of times
    ###############################################
    rm -f tmpfile
    for item in ${items[@]}
      do
      #echo "Grep ${item},${item#*_},${item%%_*}."
      grep printEventTimer ${trials[@]} | grep " ${item} " >> tmpfile # spaces are needed
      egrep "\# m2l_host" ${trials[@]} | grep ${item#*_} | grep host_${item%%_*} >> tmpfile # spaces are needed
    done
    #cat tmpfile
    cut -d ":" -f3 tmpfile | cut -d "=" -f2 > tmpfile_times
    #cat tmpfile_times
    ###############################################
    # Averaging
    ###############################################
    local nlines=$(wc -l tmpfile_times | awk '{print $1}')
    ./average ${nlines} ${ntrials} tmpfile_times tmpfile_items "printEventTimer"
    rm -f tmpfile tmpfile_times tmpfile_items
}


function average_printTimer()
{
    local trials=("$@") #array
    local ntrials=$#
    if [ $ntrials -eq 0 ] ; then
	echo "No files?"
	exit
    fi
    ###############################################
    # Create list of items from the first file
    ###############################################
    #echo "create list of items"
    #echo "There are ${ntrials} files."
    #echo ${trials[@]}
    #echo ${trials[0]}
    #cat ${trials[0]}
    grep printTimer ${trials[0]} | cut -d ":" -f2 | cut -d "=" -f1 > tmpfile_items
    #cat tmpfile_items
    items=($(cat tmpfile_items | xargs)) #array
    #echo "There are ${#items[*]} items: ${items[@]}"
    if [ ${#items[*]} -eq 0 ] ; then
	return 	# No timer is found
    fi
    ###############################################
    # Create list of times
    ###############################################
    rm -f tmpfile
    for item in ${items[@]}
      do
      #echo "Grep ${item}"
      grep printTimer ${trials[@]} | grep " ${item} " >> tmpfile # spaces are needed
    done
    #cat tmpfile
    cut -d ":" -f3 tmpfile | cut -d "=" -f2 > tmpfile_times
    #cat tmpfile_times
    ###############################################
    # Averaging
    ###############################################
    local nlines=$(wc -l tmpfile_times | awk '{print $1}')
    ./average ${nlines} ${ntrials} tmpfile_times tmpfile_items "printTimer"
    rm -f tmpfile tmpfile_times tmpfile_items
}


function average_calc_performance()
{
    local trials=("$@") #array
    local ntrials=$#
    if [ $ntrials -eq 0 ] ; then
	echo "No files?"
	exit
    fi
    ###############################################
    # Create list of items from the first file
    ###############################################
    #echo "create list of items"
    #echo "There are ${ntrials} files."
    #echo ${trials[@]}
    #echo ${trials[0]}
    #cat ${trials[0]}
    #110307 grep calc_performance ${trials[0]} | cut -d ":" -f2 | cut -d "=" -f1 > tmpfile_items
    foo=$(grep calc_performance ${trials[0]}); echo ${foo##*:} | cut -d "=" -f1 > tmpfile_items
    #cat tmpfile_items
    items=($(cat tmpfile_items | xargs)) #array
    #echo "There are ${#items[*]} items: ${items[@]}"
    if [ ${#items[*]} -eq 0 ] ; then
	return 	# No performace is found
    fi
    ###############################################
    # Create list of performances
    ###############################################
    rm -f tmpfile
    for item in ${items[@]}
      do
      #echo "Grep ${item}"
      grep calc_performance ${trials[@]} | grep " ${item} " >> tmpfile # spaces are needed
    done
    #cat tmpfile
    #110307 cut -d ":" -f3 tmpfile | cut -d "=" -f2 | sed s/"\[Gflop\/s\]"/""/ | sed s/"(sec is zero)"/"0.0"/ > tmpfile_perfs
    cut -d "=" -f2 tmpfile | sed s/"\[Gflop\/s\]"/""/ | sed s/"(sec is zero)"/"0.0"/ > tmpfile_perfs
    #cat tmpfile_perfs
    ###############################################
    # Averaging
    ###############################################
    local nlines=$(wc -l tmpfile_perfs | awk '{print $1}')
    ./average ${nlines} ${ntrials} tmpfile_perfs tmpfile_items "calc_performance"
    rm -f tmpfile tmpfile_perfs tmpfile_items
}


function average_num_pairwise_interactions_per_sec()
{
    local trials=("$@") #array
    local ntrials=$#
    if [ $ntrials -eq 0 ] ; then
	echo "No files?"
	exit
    fi
    ###############################################
    # Create list of items from the first file
    ###############################################
    foo=$(grep num_pairwise_interactions_per_sec ${trials[0]}); echo ${foo##*:} | cut -d "=" -f1 > tmpfile_items
    items=($(cat tmpfile_items | xargs)) #array
    if [ ${#items[*]} -eq 0 ] ; then
	return 	# No performace is found
    fi
    ###############################################
    # Create list of interaction/s
    ###############################################
    rm -f tmpfile
    for item in ${items[@]}
      do
      grep num_pairwise_interactions_per_sec ${trials[@]} | grep " ${item} " >> tmpfile # spaces are needed
    done
    cut -d "=" -f2 tmpfile | sed s/"\[G interaction\/s\]"/""/ > tmpfile_inters
    ###############################################
    # Averaging
    ###############################################
    local nlines=$(wc -l tmpfile_inters | awk '{print $1}')
    ./average ${nlines} ${ntrials} tmpfile_inters tmpfile_items "num_pairwise_interactions_per_sec"
    rm -f tmpfile tmpfile_inters tmpfile_items
}


function create_errfile()
{
    local head=$1
    local tail=$2
    local N=$3

    storedir=${head}${tail}

    trial0="${storedir}/${head}N${N}.err${tail}-trial0"
    if [ ! -f ${trial0} ] ; then
	echo "Skip for N=${N}"
	return
    fi

    trials="${storedir}/${head}N${N}.err${tail}-trial*"

    errfile=${head}N${N}.err${tail}

    average_printEventTimer  ${trials} >  ${errfile}
    average_printTimer       ${trials} >> ${errfile}
    average_calc_performance ${trials} >> ${errfile}
    average_num_pairwise_interactions_per_sec ${trials} >> ${errfile}
    echo "Created ${errfile}"
}

for omp in ${omp}
  do
  for n in ${n}
    do
      for d in ${d}
      do

	  tail=-n${n}-omp${omp}-d${d}

	  for N in 100 1000 10000 100000 1000000 10000000 100000000
	  do
	      echo omp=${omp} n=${n} d=${d} N=${N}
	      create_errfile ${head} ${tail} ${N}
	  done
      done
  done
done
