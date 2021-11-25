#!/bin/bash

# -modify2.tabular were created by kekka-lap-omp.sh

for omp in 1 8
  do
  for n in 4 8
    do

    cp scuda39A-lap-n${n}-omp${omp}-modify2.tabular scuda39A-lap-n${n}-omp${omp}-modify3.tabular

    echo "scuda39A-lap-n${n}-omp${omp}-modify3.tabular was created."

    for head in scuda32S scuda34S scuda38AD4
      do
      egrep "cline|total|downward1|m2l|set|get|le\\\_sw|gt\\\_sw" ${head}-lap-n${n}-omp${omp}-modify2.tabular > tmpfile
      egrep -v "setup|set2|get2|postm2l" tmpfile > ${head}-lap-n${n}-omp${omp}-modify3.tabular
      rm -f tmpfile
      cat ${head}-lap-n${n}-omp${omp}-modify3.tabular

      echo "${head}-lap-n${n}-omp${omp}-modify3.tabular was created."
    done

  done
done
