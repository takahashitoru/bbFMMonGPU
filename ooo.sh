#!/bin/bash

if [ $# -lt 2 ] ; then
    echo "usage: ./ooo.sh (same arguments as readtime-lap.sh)"
fi

READTIME=./readtime-lap.sh
if [ ! -f ${READTIME} ] ; then
    READTIME=../readtime-lap.sh
fi

head=$1
tail=$2

ooofile=${head}${tail}.xls
#100910${READTIME} ${head} ${tail} | grep -v cmidrule | grep -v KERN | sed s/"\\\PC{"/"\&"/g | sed s/"}"/""/g | sed s/"\\\\"/""/g | sed s/"%"/""/g | sed s/"&"/","/g > ${ooofile}
#${READTIME} ${head} ${tail} | grep -v cmidrule | grep -v KERN | grep -v multicolumn | sed s/"\\\PC{"/"\&"/g | sed s/"}"/""/g | sed s/"\\\\"/""/g | sed s/"%"/", "/g | sed s/"&"/","/g | sed s/"$"/""/g > ${ooofile}
${READTIME} ${head} ${tail} | grep -v cmidrule | grep -v KERN | grep -v multicolumn | grep -v Item | sed s/"\\\PC{"/"\&"/g | sed s/"}"/""/g | sed s/"\\\\"/""/g | sed s/"%"/", "/g | sed s/"&"/","/g  > ${ooofile}

lines=$(wc -l ${ooofile} | awk '{print $1}')
#echo "There are ${lines} line(s)"

echo -n "=" > tmpfile
for (( line=1 ; line<=${lines} ; line++ ))
  do
  # Check if this line has the non-empty second column or not.
  second=$(sed -n ${line},${line}p ${ooofile} | cut -d "," -f 2)
  if [ "${second}" != "" ] ; then
      #echo "Line ${line} has the non-empty second column: ${second}"
      echo -n "+\$B${line}" >> tmpfile
  fi
done
echo >> tmpfile

#100910echo -n "SUM" >> ${ooofile}
#100910for col in B C D E F G H I J K L M N O P Q R S # as many as necessary
echo -n ",SUM" >> ${ooofile}
for col in C D E F G H I J K L M N O P Q R S # as many as necessary
  do
  tmp=$(sed s/"B"/"${col}"/g tmpfile)
  echo -n ",\"${tmp}\"" >> ${ooofile}
done
echo >> ${ooofile}

#cat ${ooofile}
echo "Created ${ooofile}"


exit

