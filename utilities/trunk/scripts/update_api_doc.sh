#!/bin/sh

#####################################################################################
# script : update_api_doc.sh
# 	USER_LOGIN=$1
# 	SVN_PATH=$2
#
#####################################################################################

PARADISEO_REPOSITORY='scm.gforge.inria.fr/svn/paradiseo'
GFORGE=gforge.inria.fr
WEB_SITE_DOC_PATH='/home/groups/paradiseo/htdocs/addon/test'
MODULE_LIST="mo moeo peo"
 
EO_CVS_GETTER="-z3 -d:pserver:anonymous@eodev.cvs.sourceforge.net:/cvsroot/eodev co -P eo"

PID=$$
#Date
DATE=`/bin/date '+%Y%m%d%H%M%S'`
# create log file
SPY=$PWD/spy-update-api-doc-${PID}.${DATE}.log
touch $SPY

# PID
echo "PID : ${PID}"
echo "PID : ${PID}" >> $SPY
# Date
DAY_DATE=`/bin/date '+%Y%m%d'`
echo "DAY_DATE : ${DAY_DATE}"
echo "DAY_DATE : ${DAY_DATE}" >> $SPY
echo "DATE : ${DATE}"
echo "DATE : ${DATE}" >> ${SPY} 2>> ${SPY}
START_AT=`/bin/date '+%H:%M:%S'`
echo "START_AT : ${START_AT}"
echo "START_AT : ${START_AT}"  >> $SPY
echo >> ${SPY} 2>> ${SPY}

TEMP_ROOT_DIR=/tmp
TEMP_DIR_NAME=temp_$DATE

#error
GLOBAL_ERROR=-2
VERSION_SYNTAX_ERROR=-3

# argument
USER_LOGIN=$1
SVN_PATH=$2

if [ $# -lt 1 ]
then
	echo
        echo "=ERR=> Usage : $0 <user login> [paradiseo svn tag]"
        exit 1
fi

SVN=svn+ssh://$USER_LOGIN@$PARADISEO_REPOSITORY


#### Specific case for EO ##################################################################

# create a temporary directory
mkdir $TEMP_ROOT_DIR/$TEMP_DIR_NAME >> ${SPY} 2>> ${SPY}
if [ ! "$?" = "0" ]
then
	echo ''
	echo "	Cannot create directory $TEMP_ROOT_DIR/$TEMP_DIR_NAME"
	echo "	Cannot create directory $TEMP_ROOT_DIR/$TEMP_DIR_NAME" >> ${SPY} 2>> ${SPY}
	echo -e ' \033[40m\033[1;33m### END ### \033[0m '
	exit $GLOBAL_ERROR
fi 
echo "$TEMP_ROOT_DIR/$TEMP_DIR_NAME created" >> ${SPY} 2>> ${SPY}

# go in the last subdir
cd $TEMP_ROOT_DIR/$TEMP_DIR_NAME >> ${SPY} 2>> ${SPY}
if [ ! "$?" = "0" ]
then
	echo ''
	echo "	Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME"
	echo "	Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME" >> ${SPY} 2>> ${SPY}
	echo -e ' \033[40m\033[1;33m### END ### \033[0m '
	exit $GLOBAL_ERROR
fi 
echo "Been in $TEMP_ROOT_DIR/$TEMP_DIR_NAME" >> ${SPY} 2>> ${SPY}


# get the eo sources
cvs $EO_CVS_GETTER >> ${SPY} 2>> ${SPY}
if [ ! "$?" = "0" ]
then
	echo ''
	echo "	Cannot checkout EO from $EO_CVS_GETTER"
	echo "	Cannot checkout EO from $EO_CVS_GETTER" >> ${SPY} 2>> ${SPY}
	echo -e ' \033[40m\033[1;33m### END ### \033[0m '
	exit $GLOBAL_ERROR
fi 
echo "Checkout EO from $EO_CVS_GETTER" >> ${SPY} 2>> ${SPY}


# move eo to paradiseo-eo
mv $TEMP_ROOT_DIR/$TEMP_DIR_NAME/eo $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo >> ${SPY} 2>> ${SPY}
if [ ! "$?" = "0" ]
then
	echo ''
	echo "	Cannot rename eo to paradiseo-eo"
	echo "	Cannot rename eo to paradiseo-eo" >> ${SPY} 2>> ${SPY}
	echo -e ' \033[40m\033[1;33m### END ### \033[0m '
	exit $GLOBAL_ERROR
fi 
echo "Renamed eo to paradiseo-eo ">> ${SPY} 2>> ${SPY}


# go in the build dir
mkdir $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build >> ${SPY} 2>> ${SPY}
if [ ! "$?" = "0" ]
then
	echo ''
	echo "	Cannot create $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build"
	echo "	Cannot create $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build" >> ${SPY} 2>> ${SPY}
	echo -e ' \033[40m\033[1;33m### END ### \033[0m '
	exit $GLOBAL_ERROR
fi 
echo "Cannot create $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build" >> ${SPY} 2>> ${SPY}


# get the eo sources 
cd $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build >> ${SPY} 2>> ${SPY}
if [ ! "$?" = "0" ]
then
	echo ''
	echo "	Cannot go  in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build"
	echo "	Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build" >> ${SPY} 2>> ${SPY}
	echo -e ' \033[40m\033[1;33m### END ### \033[0m '
	exit $GLOBAL_ERROR
fi 
echo "Cannot go  in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build" >> ${SPY} 2>> ${SPY}

# Configure the module
cmake .. -G"Unix Makefiles" >> ${SPY} 2>> ${SPY}
if [ ! "$?" = "0" ]
then
	echo ''
	echo "	Cannot go in run CMake: cmake .."
	echo "	Cannot go in run CMake: cmake .." >> ${SPY} 2>> ${SPY}
	echo -e ' \033[40m\033[1;33m### END ### \033[0m '
	exit $GLOBAL_ERROR
fi 
echo "CMake run for EO with: cmake .." >> ${SPY} 2>> ${SPY}


# Go in the module's doc dir
cd $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build/doc >> ${SPY} 2>> ${SPY}
if [ ! "$?" = "0" ]
then
	echo ''
	echo " Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build/doc"
	echo " Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build/doc" >> ${SPY} 2>> ${SPY}
	echo -e ' \033[40m\033[1;33m### END ### \033[0m '
	exit $GLOBAL_ERROR
fi 
echo "Go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-eo/build/doc OK" >> ${SPY} 2>> ${SPY}
	
# Generate the doc
make doc >> ${SPY} 2>> ${SPY}
if [ ! "$?" = "0" ]
then
	echo ''
	echo "	Cannot go generate the doc for paradiseo-eo"
	echo "  Cannot go generate the doc for paradiseo-eo" >> ${SPY} 2>> ${SPY}
	echo -e ' \033[40m\033[1;33m### END ### \033[0m '
	exit $GLOBAL_ERROR
fi 
echo "Doc generated" >> ${SPY} 2>> ${SPY}
#############################################################################################




####### ParadisEO's module part ##################################################################
for MODULE in $MODULE_LIST; do
	
	# go in the last subdir
	cd $TEMP_ROOT_DIR/$TEMP_DIR_NAME >> ${SPY} 2>> ${SPY}
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME"
		echo "	Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME" >> ${SPY} 2>> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		exit $GLOBAL_ERROR
	fi 
	echo "Been in $TEMP_ROOT_DIR/$TEMP_DIR_NAME" >> ${SPY} 2>> ${SPY}


	# checkout the sources of ParadisEO from the svn repository
	if [ "$SVN_PATH" = "" ]
	then
		svn checkout $SVN/trunk $TEMP_ROOT_DIR/$TEMP_DIR_NAME >> ${SPY} 2>> ${SPY}
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "	Cannot checkout from $SVN/trunk. Make sure you can access to the repository."
			echo "	Cannot checkout from $SVN/trunk" >> ${SPY} 2>> ${SPY}
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			exit $GLOBAL_ERROR
		fi
		echo "svn checkout of $SVN/trunk DONE" >> ${SPY} 2>> ${SPY} 
	else
		svn checkout $SVN/$SVN_PATH $TEMP_ROOT_DIR/$TEMP_DIR_NAME >> ${SPY} 2>> ${SPY}
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "	Cannot checkout from $SVN/$SVN_PATH"
			echo "	Cannot checkout from $SVN/$SVN_PATH" >> ${SPY} 2>> ${SPY}
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			exit $GLOBAL_ERROR
		fi 
		echo "svn checkout of $SVN/$SVN_PATH DONE" >> ${SPY} 2>> ${SPY} 
	fi

	# Go in the module's dir
	cd $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build >> ${SPY} 2>> ${SPY}
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build"
		echo " Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build" >> ${SPY} 2>> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		exit $GLOBAL_ERROR
	fi 
	echo "Been in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build" >> ${SPY} 2>> ${SPY}


	# Configure the module
	cmake .. -G"Unix Makefiles" -Dconfig=$TEMP_ROOT_DIR/$TEMP_DIR_NAME/install.cmake >> ${SPY} 2>> ${SPY}
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot go in run CMake: cmake .. -Dconfig=$TEMP_ROOT_DIR/$TEMP_DIR_NAME/install.cmake"
		echo "	Cannot go in run CMake: cmake .. -Dconfig=$TEMP_ROOT_DIR/$TEMP_DIR_NAME/install.cmake" >> ${SPY} 2>> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		exit $GLOBAL_ERROR
	fi 
	echo "CMake run for $MODULE with: cmake .. -Dconfig=$TEMP_ROOT_DIR/$TEMP_DIR_NAME/install.cmake" >> ${SPY} 2>> ${SPY}


	# Go in the module's doc dir
	cd $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc >> ${SPY} 2>> ${SPY}
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc"
		echo " Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc" >> ${SPY} 2>> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		exit $GLOBAL_ERROR
	fi 
	echo "Been in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc" >> ${SPY} 2>> ${SPY}

	# Generate the doc
	make doc >> ${SPY} 2>> ${SPY}
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot go generate the doc for paradiseo-$MODULE"
		echo "  Cannot go generate the doc for paradiseo-$MODULE" >> ${SPY} 2>> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		exit $GLOBAL_ERROR
	fi 
	echo "Doc generated" >> ${SPY} 2>> ${SPY}

	# Make an archive of the previous on-line doc
	ssh $USER_LOGIN@$GFORGE tar -cvf doc-$MODULE-${DATE}.tgz $WEB_SITE_DOC_PATH/paradiseo-$MODULE/doc >> ${SPY} 2>> ${SPY}
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot make an archive of the previous doc for paradiseo-$MODULE in $WEB_SITE_DOC_PATH/paradiseo-$MODULE/doc"
		echo "  Cannot make an archive of the previous doc for paradiseo-$MODULE in $WEB_SITE_DOC_PATH/paradiseo-$MODULE/doc" >> ${SPY} 2>> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		exit $GLOBAL_ERROR
	fi 
	echo "Doc archive made for paradiseo-$MODULE" >> ${SPY} 2>> ${SPY}

	# Copy build/doc to source/doc
	cp -Rf $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc/* $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/doc >> ${SPY} 2>> ${SPY}
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot copy build doc $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc to source doc dir $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE"
		echo "  Cannot copy build doc $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc to source doc dir $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE" >> ${SPY} 2>> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		exit $GLOBAL_ERROR
	fi 
	echo "Copied build doc $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc to source doc dir $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE" >> ${SPY} 2>> ${SPY}
	
	# Commit the generated doc into the tag/svn path
	svn commit -m "Doc script updates API documentation module=paradiseo-$MODULE SPY=$SPY" $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/doc >> ${SPY} 2>> ${SPY}
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	svn commit -m \"Doc script updates API documentation SPY=$SPY\" $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc"
		echo "  svn commit -m \"Doc script updates API documentation SPY=$SPY\" $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc" >> ${SPY} 2>> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		exit $GLOBAL_ERROR
	fi 
	echo "Doc script updates API documentation for paradiseo-$MODULE" >> ${SPY} 2>> ${SPY}

	# Remove php scripts that cannot be copied on the gforge
	rm -f $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc/html/*.php >> ${SPY} 2>> ${SPY}
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot remove php file from $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc/html"
		echo "  Cannot remove php file from $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc/html" >> ${SPY} 2>> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		exit $GLOBAL_ERROR
	fi 
	echo "Removed php file from $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc/html" >> ${SPY} 2>> ${SPY}

	# Copy the doc to the web site
	scp $TEMP_ROOT_DIR/$TEMP_DIR_NAME/paradiseo-$MODULE/build/doc/html/* $USER_LOGIN@$GFORGE:$WEB_SITE_DOC_PATH/paradiseo-$MODULE/doc >> ${SPY} 2>> ${SPY}
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot copy the doc of paradiseo-$MODULE on the web site"
		echo "  Cannot copy the doc of paradiseo-$MODULE on the web site" >> ${SPY} 2>> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		exit $GLOBAL_ERROR
	fi 
	echo "Doc generated" >> ${SPY} 2>> ${SPY}

done


# remove the temporary directories
rm -Rf $TEMP_ROOT_DIR/$TEMP_DIR_NAME >> ${SPY} 2>> ${SPY}
if [ ! "$?" = "0" ]
then
	echo ''
	echo "	Cannot remove temp directory $TEMP_ROOT_DIR/$TEMP_DIR_NAME"
	echo "	Cannot remove temp directory $TEMP_ROOT_DIR/$TEMP_DIR_NAME" >> ${SPY} 2>> ${SPY}
	echo "Try to remove it \" by hand\" "
	echo -e ' \033[40m\033[1;33m### END ### \033[0m '
	return $GLOBAL_ERROR
fi
echo "$TEMP_ROOT_DIR/$TEMP_DIR_NAME directory completely removed" >> ${SPY} 2>> ${SPY}


echo
echo
END_AT=`/bin/date '+%H:%M:%S'`
echo "END_AT : ${START_AT}"
echo >> ${SPY} 2>> ${SPY}
echo "END_AT : ${START_AT}" >> ${SPY} 2>> ${SPY}
echo "-----------------------------------------------------------------------------------------" >> ${SPY} 2>> ${SPY}

echo
echo "Successfull doc generation, see ${SPY} for more details"
echo
exit