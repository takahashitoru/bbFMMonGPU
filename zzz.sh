#!/bin/bash

# Choose/prepare for the template of Makefile

echo "HOSTNAME=${HOSTNAME}"
if [ "${HOSTNAME}" = "q.nuem.nagoya-u.ac.jp" ] ; then
    mymake=./make_q.sh # T7500 with icc 12.0
    #mymake=./make_q_mkl.sh # a.nagoya-u.ac.jp with icc 12.0 and MKL; this does not work correctly...
elif [ ${HOSTNAME} = a ] ; then
    mymake=./make_a.sh # a.nagoya-u.ac.jp with icc 12.1
fi
#mymake=./make_icc.sh # T7500 with icc 12.0
#mymake=./make_gcc.sh # T7500 with gcc 4.4.5
#mymake=./make_certainty.sh # icc 12.0
#mymake=./make_optiplex.sh # Eric's machine

cumake=./cumake_q.sh # T7500 with icc 12.0
#cumake=./cumake_icc.sh # T7500 with icc 12.0
#cumake=./cumake_gcc.sh # T7500 with gcc 4.4.5
#cumake=./cumake_certainty.sh # icc 12.0
#cumake=./cumake_optiplex.sh # Eric's machine

function BOTH()
{
    echo "OPTS+=-D$1 CUOPTS+=-D$1"
}

clean="make clean"

$clean
$mymake EXE=sfmm9U-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9U OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9U-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9U OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9T-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9T OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9T-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9T OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9S-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9S OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9S-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9S OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9R-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9R OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9R-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9R OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9Q-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9Q OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9Q-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9Q OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9P-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9P OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9P-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9P OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9O-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9O OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9O-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9O OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9N-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9N OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9N-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9N OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9M-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9M OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9M-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9M OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9L-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9L OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9L-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9L OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9K-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9K OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9K-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9K OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9J-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9J OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9J-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9J OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9I-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9I OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9I-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9I OPTS+=-DFAST_HOST_CODE #120214
exit


$clean
$mymake EXE=sfmm9H-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9H OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9H-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9H OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm8A-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU8 OPTS+=-DCPU8A OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120208
exit
#$clean
#$mymake EXE=dfmm8A-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU8 OPTS+=-DCPU8A OPTS+=-DFAST_HOST_CODE #120208
#exit

$clean
$mymake EXE=sdir46-lap.out OPTS+=-DLAPLACIAN OPTS+=-DENABLE_DIRECT OPTS+=-DSINGLE #120208
#exit
$clean
$mymake EXE=ddir46-lap.out OPTS+=-DLAPLACIAN OPTS+=-DENABLE_DIRECT #120208
exit



$clean
$mymake EXE=sfmm9H-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9H OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9H-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9H OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9G-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9G OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9G-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9G OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9F-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9F OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9F-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9F OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm8A-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU8 OPTS+=-DCPU8A OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120208
exit
$clean
$mymake EXE=dfmm8A-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU8 OPTS+=-DCPU8A OPTS+=-DFAST_HOST_CODE #120208
exit





$clean
$mymake EXE=sfmm9E-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9E OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9E-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9E OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9D-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9D OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9D-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9D OPTS+=-DFAST_HOST_CODE #120214
exit


$clean
$mymake EXE=sfmm9B-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9B OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9B-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9B OPTS+=-DFAST_HOST_CODE #120214
exit

$clean
$mymake EXE=sfmm9C-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9C OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9C-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9C OPTS+=-DFAST_HOST_CODE #120214
exit



$clean
$mymake EXE=sfmm9A-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9A OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120214
exit
$clean
$mymake EXE=dfmm9A-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU9 OPTS+=-DCPU9A OPTS+=-DFAST_HOST_CODE #120214
exit



for sw in 2 #default is 2
do
for nr in 4 #default is 4
do
for tpb in 64 #default is 64
do
for ss in 64 #default is 64
do
    $clean
    $cumake CUEXE=scuda46Q_ij_sw${sw}nr${nr}tpb${tpb}ss${ss}-lap.out $(BOTH LAPLACIAN) CUOPTS+=-DCUDA_VER46 CUOPTS+=-DCUDA_VER46Q CUOPTS+=-DFAST_HOST_CODE CUOPTS+=-DSCHEME=3 CUOPTS+="-DLEVEL_SWITCH_IJ=${sw}" CUOPTS+="-DNUM_ROW_GROUPS_IJ=${nr}" CUOPTS+="-DDIRECT_NUM_THREADS_PER_BLOCK=${tpb}" CUOPTS+="-DDIRECT_SHARE_SIZE=${ss}" $(BOTH SINGLE) #120208 ij
    $clean
    $cumake CUEXE=dcuda46Q_ij_sw${sw}nr${nr}tpb${tpb}ss${ss}-lap.out $(BOTH LAPLACIAN) CUOPTS+=-DCUDA_VER46 CUOPTS+=-DCUDA_VER46Q CUOPTS+=-DFAST_HOST_CODE CUOPTS+=-DSCHEME=3 CUOPTS+="-DLEVEL_SWITCH_IJ=${sw}" CUOPTS+="-DNUM_ROW_GROUPS_IJ=${nr}" CUOPTS+="-DDIRECT_NUM_THREADS_PER_BLOCK=${tpb}" CUOPTS+="-DDIRECT_SHARE_SIZE=${ss}" #120208 ij
done
done
done
done
exit


