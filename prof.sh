#!/bin/sh

#ulimit -s unlimited

function GREP_DIRECT(){
    grep 'method\|direct' $1
}

function GREP_M2L(){
    grep 'method\|m2l' $1
}

function GREP_UXP(){
    grep 'method\|uxp' $1
}

function GREP_MERGE(){
    grep 'method\|merge' $1
}

if [ $# -lt 2 ]; then
    echo "usage: ./prof.sh (exec_file) [parameters of exec_file]"
    exit
fi

exe=$1
N=$2
L=$3
n=$4
if [ $# -ge 5 ]; then
    l=$5
else
    l=-1
fi

echo "exe=$exe N=$N L=$L n=$n l=$l"

export CUDA_PROFILE=1
export CUDA_PROFILE_CSV=1
export CUDA_PROFILE_LOG=tmptmptmp
export CUDA_PROFILE_CONFIG=cuda_profile_config.txt

rm -f tmptmptmp?

echo "1st..."
echo -e "gld_coherent\n gld_incoherent\n gst_coherent\n gst_incoherent" > $CUDA_PROFILE_CONFIG
(/usr/bin/time $exe $N $L $n $l > pout) >& perr
GREP_DIRECT $CUDA_PROFILE_LOG | cut -d"," -f1,2,3,4,5,6,7,8 > tmptmptmp1
GREP_M2L $CUDA_PROFILE_LOG | cut -d"," -f1,2,3,4,5,6,7,8 >> tmptmptmp1
GREP_UXP $CUDA_PROFILE_LOG | cut -d"," -f1,2,3,4,5,6,7,8 >> tmptmptmp1
GREP_MERGE $CUDA_PROFILE_LOG | cut -d"," -f1,2,3,4,5,6,7,8 >> tmptmptmp1
echo "2nd..."
echo -e "local_load\n local_store\n branch\n divergent_branch" > $CUDA_PROFILE_CONFIG
(/usr/bin/time $exe $N $L $n $l > pout) >& perr
GREP_DIRECT $CUDA_PROFILE_LOG | cut -d"," -f5,6,7,8 > tmptmptmp2
GREP_M2L $CUDA_PROFILE_LOG | cut -d"," -f5,6,7,8 >> tmptmptmp2
GREP_UXP $CUDA_PROFILE_LOG | cut -d"," -f5,6,7,8 >> tmptmptmp2
GREP_MERGE $CUDA_PROFILE_LOG | cut -d"," -f5,6,7,8 >> tmptmptmp2
echo "3rd..."
echo -e "instructions\n warp_serialize\n cta_launched" > $CUDA_PROFILE_CONFIG
(/usr/bin/time $exe $N $L $n $l > pout) >& perr
GREP_DIRECT $CUDA_PROFILE_LOG | cut -d"," -f5,6,7 > tmptmptmp3
GREP_M2L $CUDA_PROFILE_LOG | cut -d"," -f5,6,7 >> tmptmptmp3
GREP_UXP $CUDA_PROFILE_LOG | cut -d"," -f5,6,7 >> tmptmptmp3
GREP_MERGE $CUDA_PROFILE_LOG | cut -d"," -f5,6,7 >> tmptmptmp3

#proffile=prof.result
tmp=${exe%.out}; head=${tmp#*/}
if [ $l -eq -1 ] ; then
    proffile=${head}N${N}n${n}.prof
else
    proffile=${head}N${N}n${n}l${l}.prof
fi
rm -f $proffile
lines=`wc -l tmptmptmp1 | cut -d" " -f1`
echo "lines -> $lines"
for (( l = 1; l <= $lines; l++ ))
  do
  echo `head -$l tmptmptmp1 | tail -1`,`head -$l tmptmptmp2 | tail -1`,`head -$l tmptmptmp3 | tail -1` >> $proffile
done
rm -f tmptmptmp?
cat $proffile
echo $proffile was created.
echo "done."

    


