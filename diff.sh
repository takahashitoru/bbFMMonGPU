#!/bin/csh
#
# This script shows the differeces between current version source codes and 
# the previous ones. Execute by makefile as "make diff", or directly 
#
#     diff.sh   ../0404    foo1.f foo2.f foo3.f .....
#            (1)^^^^^^^ (2)^^^^^^^^^^^^^^^^^^^^^^^
#
# where the 1st argument means the directory where the previous version 
# exists and the other arguments mean the source codes that you want to
# take difference. Thus, the above exapmle means
#     diff.sh foo1.f ../0404/foo1.f
#     diff.sh foo2.f ../0404/foo2.f
#     diff.sh foo3.f ../0404/foo3.f 
# and so on.  
#
# 2000.04.07
###############################################################################
if ($#argv == "0") then
	echo diff.sh: invalid arguments
	exit(1)
endif

set OLD = $argv[1]

echo previous version exists at $OLD

set N = $#argv
set i = 2
while ($i <= "$N")
	echo "@" $argv[$i]
	diff $OLD/$argv[$i] $argv[$i]
	@ i++
end
