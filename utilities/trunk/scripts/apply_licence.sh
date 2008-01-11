#!/bin/sh


#
# check the args
#
if [ $# -lt 3 ]
then
	echo "Use: apply_licence <dir or file> <licence content file> <authors>"
	exit 1
fi 


if [ -d $1 ]
then
	TARGETS=ls $1
	echo " Apply the licence for all files located in $1"
else
	if [ ! -f $1 ]
	then
		echo " Error: unknown file \"$1\" "
		exit 1
	else
		TARGETS=$1
	fi
fi


if [ ! -f $2 ]
then
	echo " Error: unknown licence file \"$2\" "
	exit 1
else
	LICENCE_CONTENT_FILE=$2
fi


# manage the authors
AUTHORS=$3

# remove the old licence if there's one
for i in $TARGETS 
do
	echo " Removing the licence of $i"
	total=0
	lc=0
	keep=0

	while read alline 
	do
		lc=`expr $lc + 1`
	done < $i

	while read aline && $CONTINUE
	do
		if [ "$aline" = "*/" ] 
		then		
			keep=`expr $lc - $total`
			tail -n $keep > $i.tmp
			mv $i.tmp $i
			break
		else 
			total=`expr $total + 1`
		fi
	done < $i
done


# get the reverse licence file whose lines will be inserted at the begining of each target
tac $LICENCE_CONTENT_FILE > $LICENCE_CONTENT_FILE.reverse.tmp

# loop over the file list
for j in $TARGETS 
do 
	echo " Inserting the new licence in $j"
	while read line
	do
		if [ "$line" = "NAMES" ] 
		then
			sed -i "1i * $AUTHORS" $j
		else 
			sed -i "1i $line" $j
		fi
	done < $LICENCE_CONTENT_FILE.reverse.tmp

	# insert the file name: <file.type>
	sed -i "1i  * <$j>" $j
	sed -i "1i /* " $j
done


# remove the reverse tmp file
rm -f $LICENCE_CONTENT_FILE.reverse.tmp

exit 0


