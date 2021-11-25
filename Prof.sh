#!/bin/bash
#PBS -N Prof
#PBS -j oe
#PBS -q tesla
cd ~/stanford/090629/work
#../prof.sh ../scuda38G2.out 1000000 1 8
../prof.sh ../scuda38G2.out 10000000 1 8
