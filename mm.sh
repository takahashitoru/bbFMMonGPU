#!/bin/bash

if [ $# -lt 1 ] ; then
    echo "usage: ./mm.sh (ver)" # (ver) must be specified in Makefile.cuda
    exit
fi

ver=$1
exe=./a${ver}.out

make -f Makefile.cuda $ver TESTEXE=$exe

function GREP_M2L(){
    grep 'method\|m2l' $1
}


export CUDA_PROFILE=1
export CUDA_PROFILE_CSV=1
export CUDA_PROFILE_LOG=tmptmptmp
export CUDA_PROFILE_CONFIG=cuda_profile_config.txt

rm -rf tmptmptmp?

outfile=pout$ver
errfile=perr$ver

echo "1st..."
echo -e "gld_coherent\n gld_incoherent\n gst_coherent\n gst_incoherent" > $CUDA_PROFILE_CONFIG
(/usr/bin/time $exe > $outfile ) >& $errfile 
grep err $errfile
GREP_M2L $CUDA_PROFILE_LOG | cut -d"," -f1,2,3,4,5,6,7,8 >> tmptmptmp1
echo "2nd..."
echo -e "local_load\n local_store\n branch\n divergent_branch" > $CUDA_PROFILE_CONFIG
(/usr/bin/time $exe > $outfile ) >& $errfile 
grep err $errfile
GREP_M2L $CUDA_PROFILE_LOG | cut -d"," -f5,6,7,8 >> tmptmptmp2
echo "3rd..."
echo -e "instructions\n warp_serialize\n cta_launched" > $CUDA_PROFILE_CONFIG
(/usr/bin/time $exe > $outfile ) >& $errfile 
grep err $errfile
GREP_M2L $CUDA_PROFILE_LOG | cut -d"," -f5,6,7 >> tmptmptmp3

tmp=${exe%.out}; head=${tmp#*/}
proffile=${head}.prof

rm -f $proffile
lines=`wc -l tmptmptmp1 | cut -d" " -f1`
echo "lines -> $lines"
for (( l = 1; l <= $lines; l++ ))
  do
  echo `head -$l tmptmptmp1 | tail -1`,`head -$l tmptmptmp2 | tail -1`,`head -$l tmptmptmp3 | tail -1` >> $proffile
done
cat $proffile
echo $proffile was created.
echo "done."

    




