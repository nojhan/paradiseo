#!/bin/bash

lines=0


for i in `ls $1`  
do
	mem=$PWD
	if [ ! -d $i ] 
	then
		tmp=`wc -l $i | cut -f1 -d\ `
		echo $tmp >> /tmp/count.txt
	else
		cd $i
		/home/tlegrand/OPAC/software/paradisEO/repository/paradiseo/trunk/count_lines.sh $PWD
		cd $mem
	fi
done
