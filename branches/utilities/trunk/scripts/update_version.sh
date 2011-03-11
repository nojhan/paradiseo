#!/bin/sh

#####################################################################################
# script : update_version.sh
# 	update a version of paradiseo_full_package from SVN repository source
#
#####################################################################################

#------------------------------------------------------#
#-- FUNCTION   :  mail		                    ---#
#------------------------------------------------------#
#-- PARAMETERS :  				    ---#
#--	$1 : sujet                                  ---#
#--	$2 : objet	                            ---#
#--	$3 : corps	                            ---#
#--                                                 ---#
#------------------------------------------------------#
#-- CODE RETURN : 0                                 ---#
#------------------------------------------------------#
function send_mail
{
	MAIL_SUJET=$1
	MAIL_MESSAGE=$2
	MAIL_DEST=$3
	for DEST in ${MAIL_DEST}
	do
	# build Mail
	/usr/bin/mailx -s "${MAIL_SUJET}" ${DEST} << EOF
${MAIL_MESSAGE}
EOF
	done	
	return 0
}

# variables
ARCHIVE_TARGET_PATH=.
TEMP_ROOT_DIR=/tmp

EO_REPOSITORY='eodev.cvs.sourceforge.net'
EO_SHARED_MODULE_PATH='/cvsroot/eodev'
EO_REPOSITORY_CONNECTION_MODE='pserver'
EO_REPOSITORY_LOGIN='anonymous'
EO_MODULE_NAME='eo'

PARADISEO_REPOSITORY='scm.gforge.inria.fr/svn/paradiseo'

PACKAGE_SUFFIX_TAR_BZ2='bz2'
PACKAGE_SUFFIX_TAR_GZ='tar.gz'
PACKAGE_SUFFIX_ZIP='zip'
TAR_BZ2_OPTIONS='cjvf'
TAR_GZ_OPTIONS='cvzf'

PARADISEO_ARCHIVE_DOWNLOAD_SITE=duff.lifl.fr
PARADISEO_ARCHIVE_DOWNLOAD_ADDRESS=/home/www/LIFL/htdocs/Equipes/OPAC/Paradiseo/download

#error
GLOBAL_ERROR=-2
VERSION_SYNTAX_ERROR=-3

# argument
USER_LOGIN=$1
PACKAGE_NAME=$2
PACKAGE_VERSION=$3
SVN_PATH=$4
EO_CVS_TAG=$5

PID=$$
#Date
DATE=`/bin/date '+%Y%m%d%H%M%S'`
# create log file
SPY=$PWD/spy-${PID}.${DATE}.log
touch $SPY

# PID
echo "PID : ${PID}"
echo "PID : ${PID}" >> $SPY
# Date
DAY_DATE=`/bin/date '+%Y%m%d'`
echo "DAY_DATE : ${DAY_DATE}"
echo "DAY_DATE : ${DAY_DATE}" >> $SPY
echo "DATE : ${DATE}"
echo "DATE : ${DATE}" >> ${SPY}
START_AT=`/bin/date '+%H:%M:%S'`
echo "START_AT : ${START_AT}"
echo "START_AT : ${START_AT}"  >> $SPY
echo >> ${SPY}


# check the number of parameters
if [ $# -lt 3 ]
then
	echo
        echo "=ERR=> Usage : $0 <user login> <package name> <version> [paradiseo svn tag] [eo cvs tag]"
        exit 1
fi

# check version syntax
function check_version()
{
	VERSION=$1
	VERSION_SYNTAX="`echo ${VERSION} | /bin/grep [0-9].[0-9]`"
	echo "VERSION : $VERSION_SYNTAX" >> ${SPY}
	if [ "$VERSION_SYNTAX" = "" ]
	then
		return $VERSION_SYNTAX_ERROR
	else
		return 0
	fi	
}

# get last sources
function build_archive()
{
	GETBACK=$PWD
	SVN=svn+ssh://$1@$PARADISEO_REPOSITORY
	TEMP_DIR_NAME=temp_$DATE

	# create a temporary directory
	mkdir $TEMP_ROOT_DIR/$TEMP_DIR_NAME/
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot create directory $TEMP_ROOT_DIR/$TEMP_DIR_NAME"
		echo "	Cannot create directory $TEMP_ROOT_DIR/$TEMP_DIR_NAME" >> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $GLOBAL_ERROR
	fi 
	echo "$TEMP_ROOT_DIR/$TEMP_DIR_NAME created" >> ${SPY}

	# create a subdir with the full name of the release
	mkdir $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot create directory $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION"
		echo "	Cannot create directory $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION" >> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $GLOBAL_ERROR
	fi 
	echo "$TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION created" >> ${SPY}
	

	# go in the last subdir
	cd $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION"
		echo "	Cannot go in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION" >> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $GLOBAL_ERROR
	fi 
	echo "Been in $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION" >> ${SPY}


	# checkout the sources of EO from the cvs repository within an anonymous access (ssh mode)
	# Always extract the source from the HEAD cvs tag.
	if [ "$CVS_PATH" = "" ]
	then 
		cvs -z3  -d:$EO_REPOSITORY_CONNECTION_MODE:$EO_REPOSITORY_LOGIN@$EO_REPOSITORY:$EO_SHARED_MODULE_PATH checkout -r$EO_CVS_TAG $EO_MODULE_NAME

		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "	Cannot checkout EO sources from $EO_REPOSITORY:$EO_SHARED_MODULE_PATH from tag $EO_CVS_TAG"
			echo "	Cannot checkout EO sources from$EO_REPOSITORY:$EO_SHARED_MODULE_PATH from tag $EO_CVS_TAG" >> ${SPY}
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $GLOBAL_ERROR
		fi
		echo "cvs checkout of $EO_REPOSITORY:$EO_SHARED_MODULE_PATH DONE from tag $EO_CVS_TAG" >> ${SPY} 
	fi

	
	# remove all the ".cvs" config directories
	for i in `find $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION -name CVS -type d`; do 
		rm -Rf $i;
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "	Cannot remove $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION/$i"
			echo "	Cannot remove $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION/$i" >> ${SPY}
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $GLOBAL_ERROR
		fi
		echo "$i removed from final archive" >> ${SPY} 
	 done

	# remove all the ".cvsignore" config directories
	for i in `find $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION -name .cvs* -type f`; do 
		rm -Rf $i;
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "	Cannot remove $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION/$i"
			echo "	Cannot remove $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION/$i" >> ${SPY}
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $GLOBAL_ERROR
		fi
		echo "$i removed from final archive" >> ${SPY} 
	 done

	# move eo --> paradiseo-eo
	mv $EO_MODULE_NAME paradiseo-$EO_MODULE_NAME
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot move $EO_MODULE_NAME to paradiseo-$EO_MODULE_NAME"
		echo "	Cannot move $EO_MODULE_NAME to paradiseo-$EO_MODULE_NAME" >> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $GLOBAL_ERROR
	fi 
	echo "$EO_MODULE_NAME moved to paradiseo-$EO_MODULE_NAME" >> ${SPY}


	# checkout the sources of ParadiEO from the svn repository
	if [ "$SVN_PATH" = "" ]
	then
		svn checkout $SVN/trunk $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "	Cannot checkout from $SVN/trunk. Make sure you can access to the repository."
			echo "	Cannot checkout from $SVN/trunk" >> ${SPY}
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $GLOBAL_ERROR
		fi
		echo "svn checkout of $SVN/trunk DONE" >> ${SPY} 
	else
		svn checkout $SVN/$SVN_PATH $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "	Cannot checkout from $SVN/$SVN_PATH"
			echo "	Cannot checkout from $SVN/$SVN_PATH" >> ${SPY}
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $GLOBAL_ERROR
		fi 
		echo "svn checkout of $SVN/$SVN_PATH DONE" >> ${SPY} 
	fi

	# remove all the ".svn" config directories
	for i in `find $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION -name \.svn -type d`; do 
		rm -Rf $i;
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "	Cannot remove $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION/$i"
			echo "	Cannot remove $TEMP_ROOT_DIR/$TEMP_DIR_NAME/$PACKAGE_NAME-$PACKAGE_VERSION/$i" >> ${SPY}
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $GLOBAL_ERROR
		fi
		echo "$i removed from final archive" >> ${SPY} 
	 done
	
	cd $TEMP_ROOT_DIR/$TEMP_DIR_NAME 


	# create .tar.bz2 archive
	tar $TAR_BZ2_OPTIONS $2-$3.$PACKAGE_SUFFIX_TAR_BZ2 $PACKAGE_NAME-$PACKAGE_VERSION
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot create archive with \" tar -$TAR_OPTIONS $2-$3.$PACKAGE_SUFFIX_TAR_BZ2 $PACKAGE_NAME-$PACKAGE_VERSION \" "
		echo "	Cannot create archive with \" tar -$TAR_OPTIONS $2-$3.$PACKAGE_SUFFIX_TAR_BZ2 $PACKAGE_NAME-$PACKAGE_VERSION \" " >> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $GLOBAL_ERROR
	fi 
	echo "$2-$3.$PACKAGE_SUFFIX_TAR_BZ2 archive created" >> ${SPY}


	# move the archive in the initial directory
	mv $2-$3.$PACKAGE_SUFFIX_TAR_BZ2 $GETBACK
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot move archive from $PWD to $GETBACK"
		echo "	Cannot move archive from $PWD to $GETBACK" >> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $GLOBAL_ERROR
	fi 
	echo "$2-$3.$PACKAGE_SUFFIX_TAR_BZ2 moved from $PWD to  $GETBACK " >> ${SPY}


	#create tar.gz archive
	tar $TAR_GZ_OPTIONS $2-$3.$PACKAGE_SUFFIX_TAR_GZ $PACKAGE_NAME-$PACKAGE_VERSION
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot create archive with \" tar -$TAR_OPTIONS $2-$3.$PACKAGE_SUFFIX_TAR_GZ $PACKAGE_NAME-$PACKAGE_VERSION \" "
		echo "	Cannot create archive with \" tar -$TAR_OPTIONS $2-$3.$PACKAGE_SUFFIX_TAR_GZ $PACKAGE_NAME-$PACKAGE_VERSION \" " >> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $GLOBAL_ERROR
	fi 
	echo "$2-$3.$PACKAGE_SUFFIX_TAR_GZ archive created" >> ${SPY}


	# move the archive in the initial directory
	mv $2-$3.$PACKAGE_SUFFIX_TAR_GZ $GETBACK
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot move archive from $PWD to $GETBACK"
		echo "	Cannot move archive from $PWD to $GETBACK" >> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $GLOBAL_ERROR
	fi 
	echo "$2-$3.$PACKAGE_SUFFIX_TAR_GZ moved from $PWD to $GETBACK " >> ${SPY}


	#create zip archive
	zip -r $2-$3.$PACKAGE_SUFFIX_ZIP $PACKAGE_NAME-$PACKAGE_VERSION
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot create archive with \" zip -r $2-$3.$PACKAGE_SUFFIX_ZIP $PACKAGE_NAME-$PACKAGE_VERSION \" "
		echo "	Cannot create archive with \" zip -r $2-$3.$PACKAGE_SUFFIX_ZIP $PACKAGE_NAME-$PACKAGE_VERSION\" " >> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $GLOBAL_ERROR
	fi 
	echo "$2-$3.$PACKAGE_SUFFIX_ZIP archive created" >> ${SPY}


	# move the archive in the initial directory
	mv $2-$3.$PACKAGE_SUFFIX_ZIP $GETBACK
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot move archive from $PWD to $GETBACK"
		echo "	Cannot move archive from $PWD to $GETBACK" >> ${SPY}
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $GLOBAL_ERROR
	fi 
	echo "$2-$3.$PACKAGE_SUFFIX_ZIP moved from $PWD to $GETBACK " >> ${SPY}


	# come back where we were at the beginning
	cd $GETBACK


	# remove the temporary directories
	rm -Rf $TEMP_ROOT_DIR/$TEMP_DIR_NAME
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Cannot remove temp directory $TEMP_ROOT_DIR/$TEMP_DIR_NAME"
		echo "	Cannot remove temp directory $TEMP_ROOT_DIR/$TEMP_DIR_NAME" >> ${SPY}
		echo "Try to remove it \" by hand\" "
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $GLOBAL_ERROR
	fi
	echo "$TEMP_ROOT_DIR/$TEMP_DIR_NAME directory completely removed" >> ${SPY}

# 	# need to send the archive on the website ?
# 	echo
# 	echo "Do you want to send the archive $PACKAGE_NAME-$PACKAGE_VERSION.$PACKAGE_SUFFIX on $PARADISEO_ARCHIVE_DOWNLOAD_SITE:$PARADISEO_ARCHIVE_DOWNLOAD_ADDRESS as $PARADISEO_ARCHIVE_DOWNLOAD_NAME ? (Y/n) "
# 	while :
# 	do
# 		read answer
# 		if [ "$answer" = "Y" ]
# 		then
# 			echo "Please give your login to connect to $PARADISEO_ARCHIVE_DOWNLOAD_SITE"
# 			read login
# 			cp $PACKAGE_NAME-$PACKAGE_VERSION.$PACKAGE_SUFFIX $PARADISEO_ARCHIVE_DOWNLOAD_NAME
# 			scp  $PARADISEO_ARCHIVE_DOWNLOAD_NAME $login@$PARADISEO_ARCHIVE_DOWNLOAD_SITE:$PARADISEO_ARCHIVE_DOWNLOAD_ADDRESS
# 			rm $PARADISEO_ARCHIVE_DOWNLOAD_NAME
# 			echo | tee -a ${SPY}
# 			echo "=> Archive sent to $PARADISEO_ARCHIVE_DOWNLOAD_SITE:$PARADISEO_ARCHIVE_DOWNLOAD_ADDRESS " | tee -a ${SPY}
# 	
# 			exit 0
# 		fi
# 		if [ "$answer" = "n" ]
# 		then
# 			exit 0
# 		fi
# 	done
	return 0
}

# check version
check_version $PACKAGE_VERSION
if [ ! "$?" = 0 ]
then
	echo
	echo "Invalid version syntax:$PACKAGE_VERSION " | tee -a ${SPY}
	echo "A valid syntax is X-Y ([0-9]-[0-9]) "
	echo
	exit
fi

# get last sources from SVN repository and build full archive
build_archive $USER_LOGIN $PACKAGE_NAME $PACKAGE_VERSION $SVN_PATH

echo
echo
END_AT=`/bin/date '+%H:%M:%S'`
echo "END_AT : ${START_AT}"
echo >> ${SPY}
echo "END_AT : ${START_AT}" >> ${SPY}
echo "-----------------------------------------------------------------------------------------" >> ${SPY}
