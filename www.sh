#!/bin/bash

########################################
# Set templates
########################################
mymake=./make_gcc.sh
#mymake=./make_icc.sh

cumake=./cumake_gcc.sh
#cumake=./cumake_icc.sh

function BOTH()
{
    echo "OPTS+=-D$1 CUOPTS+=-D$1"
}

clean="make clean"

########################################
# Compile CPU codes
########################################
$clean
$mymake EXE=sfmm8A-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU8 OPTS+=-DCPU8A OPTS+=-DFAST_HOST_CODE OPTS+=-DSINGLE #120208
#exit
$clean
$mymake EXE=dfmm8A-lap.out $(BOTH LAPLACIAN) OPTS+=-DCPU8 OPTS+=-DCPU8A OPTS+=-DFAST_HOST_CODE #120208
#exit

########################################
# Compile direct codes
########################################
$clean
$mymake EXE=sdir46-lap.out OPTS+=-DLAPLACIAN OPTS+=-DENABLE_DIRECT OPTS+=-DSINGLE #120208
#exit
$clean
$mymake EXE=ddir46-lap.out OPTS+=-DLAPLACIAN OPTS+=-DENABLE_DIRECT #120208
#exit

########################################
# Compile GPU codes (this takes a time)
########################################

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


