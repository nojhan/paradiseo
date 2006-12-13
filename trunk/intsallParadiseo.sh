#!/bin/sh

#########################################################################################
#
#	Project		:	paradisEO full package install
#	File		:	installParadiseo
#	Comment	:	# This file attempts to install ParadisEO #
#
#########################################################################################

# global variables
installStartTime=$SECONDS
resourceKitPath=$PWD
TAR_MSG=" "


#others
PARADISEO_EO_ARCHIVE=paradiseo-eo.tgz
PARADISEO_MO_ARCHIVE=paradiseo-mo.tgz	
PARADISEO_MOEO_ARCHIVE=paradiseo-moeo.tgz
PARADISEO_PEO_ARCHIVE=paradiseo-peo.bz2
LIBXML2_ARCHIVE=libxml2-2.6.0.tar.bz2
MPICH2_ARCHIVE=mpich2-1.0.3.tar.gz

# errors
SUCCESSFUL_PARADISEO_INSTALL=0
EO_UNPACKING_ERROR=100
MO_UNPACKING_ERROR=101
MOEO_UNPACKING_ERROR=102
PARADISEO_UNPACKING_ERROR=103
LIBXML_UNPACKING_ERROR=104
MPICH_UNPACKING_ERROR=105
EO_INSTALL_ERROR=106
MO_INSTALL_ERROR=107
MOEO_INSTALL_ERROR=108
PARADISEO_INSTALL_ERROR=110
LIBXML_INSTALL_ERROR=111
MPICH_INSTALL_ERROR=112
REMOVE_TEMP_DIRECTORY_ERROR=113
VAR_CONFIG_ERROR=114
MPD_COPY_ERROR=115

#------------------------------------------------------#
#-- FUNCTION   :  run_install			    ---#
#------------------------------------------------------#
#-- PARAMETERS :  				    ---#
#-- 	install path	                            ---#
#-- Major function for install                      ---#
#------------------------------------------------------#
#-- RETURN: 0 if install OK                         ---#
#------------------------------------------------------#

function run_install()
{
        installKitPath=$1

	########## STEP 0 : introduction ##########
	clear
	echo ""
	echo -e ' \033[40m\033[1;33m### ParadisEO install starting .... ### \033[0m '
	#sleep 4
	echo
	echo "Installing the environment for paradiseo... this may take about ten minutes to complete. Note that the librairies \"libxml2\" ans \"mpich2\" required for ParadisEO are provided with this package."
	sleep 3

	echo
	echo 

	########## STEP 1: unpacking paradiseo-eo ##########
	echo -e '	\033[40m\033[1;34m# STEP 1 \033[0m '
	echo '		--> Unpacking Paradiseo-EO (Evolving Objects) ...'

	tar xvf $resourceKitPath/$PARADISEO_EO_ARCHIVE --directory $installKitPath >> $TAR_MSG
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when unpacking Paradiseo-EO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $EO_UNPACKING_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 1 OK \033[0m'
		echo
	fi 
	
	########## STEP 2: unpacking paradiseo-mo ##########
	echo -e '	\033[40m\033[1;34m# STEP 2 \033[0m '
	echo '		--> Unpacking Paradiseo-MO (Moving Objects) ...'
	
	tar xvf $resourceKitPath/$PARADISEO_MO_ARCHIVE --directory $installKitPath >> $TAR_MSG
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when unpacking Paradiseo-MO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MO_UNPACKING_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 2 OK \033[0m'
		echo
	fi 

	########## STEP 3: unpacking paradiseo-moeo ##########
	echo -e '	\033[40m\033[1;34m# STEP 3 \033[0m '
	echo '		--> Unpacking Paradiseo-MOEO (Multi objective Evolving Objects) ...'
	
	tar xvf $resourceKitPath/$PARADISEO_MOEO_ARCHIVE --directory $installKitPath >> $TAR_MSG
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when unpacking Paradiseo-MOEO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MOEO_UNPACKING_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 3 OK \033[0m'
		echo
	fi 

	########## STEP 4: unpacking  paradiseo-peo ##########
	echo -e  '	\033[40m\033[1;34m# STEP 4 \033[0m '
	echo '		--> Unpacking Paradiseo-PEO (ParadisEO on dedicated parallel and/or distributed environments) ...'
	
	tar xvjf $resourceKitPath/$PARADISEO_PEO_ARCHIVE --directory $installKitPath >> $TAR_MSG
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when unpacking Paradiseo-PEO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $PARADISEO_UNPACKING_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 4 OK \033[0m'
		echo
	fi 

	########## STEP 5: unpacking libxml2 ##########
	echo -e  '	\033[40m\033[1;34m# STEP 5 \033[0m '
	echo '		--> Unpacking libxml2 (required for ParadisEO) ...'
	
	tar xvjf $resourceKitPath/$LIBXML2_ARCHIVE  --directory $installKitPath >> $TAR_MSG
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when unpacking libxml2"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $LIBXML_UNPACKING_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 5 OK \033[0m'
		echo
	fi 

	########## STEP 6: unpacking mpich2 ##########
	echo -e  '	\033[40m\033[1;34m# STEP 6 \033[0m '
	echo '		--> Unpacking mpich2 (required for ParadisEO) ...'
	
	tar xzvf $resourceKitPath/$MPICH2_ARCHIVE --directory $installKitPath >> $TAR_MSG
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when unpacking mpich2"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MPICH_UNPACKING_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 6 OK \033[0m'
		echo
	fi 

	########## STEP 7: installing paradiseo-eo ##########
	echo -e  '	\033[40m\033[1;34m# STEP 7 \033[0m '
	echo '		--> Installing Paradiseo-EO ...'
	
	cd $installKitPath/paradiseo-eo && ./autogen.sh && ./configure && make
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when installing Paradiseo-EO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $EO_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 7 OK \033[0m'
		echo
	fi 

	########## STEP 8: installing paradiseo-mo ##########
	echo -e  '	\033[40m\033[1;34m# STEP 8 \033[0m '
	echo '		--> Installing Paradiseo-MO ...'
	
	cd $installKitPath/paradiseo-mo && ./autogen.sh --with-EOdir=$installKitPath/paradiseo-eo && make
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when installing Paradiseo-MO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MO_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 8 OK \033[0m'
		echo
	fi 

	########## STEP 9: installing MOEO ##########
	echo -e  '	\033[40m\033[1;34m# STEP 9 \033[0m '
	echo '		--> Installing Paradiseo-MOEO ...'
	
	cd $installKitPath/paradiseo-moeo && ./autogen.sh --with-EOdir=$installKitPath/paradiseo-eo/ --with-MOdir=$installKitPath/paradiseo-mo/ && make
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when installing Paradiseo-MOEO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MOEO_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 9 OK \033[0m'
		echo
	fi 

	########## STEP 10: installing LIBXML2 ##########
	echo -e  '	\033[40m\033[1;34m# STEP 10 \033[0m '
	echo '		--> Installing LIBXML2 ...'
	
	mkdir $installKitPath/libxml2 && cd $installKitPath/libxml2-2.6.0/ && ./configure --prefix=$installKitPath/libxml2/ --exec-prefix=$installKitPath/libxml2/ && make && make install
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when installing LIBXML2"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $LIBXML_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 10 OK \033[0m'
		echo
	fi 


	########## STEP 11: installing MPICH2 ##########
	echo -e  '	\033[40m\033[1;34m# STEP 11 \033[0m '
	echo '		--> Installing MPICH2 ...'
	
	mkdir $installKitPath/mpich2 && cd $installKitPath/mpich2-1.0.3/ && ./configure --prefix=$installKitPath/mpich2/ && make && make install
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when installing MPICH2"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MPICH_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 11 OK \033[0m'
		echo
	fi 

	########## STEP 12: removing temp directory for libxml ##########
	echo -e  '	\033[40m\033[1;34m# STEP 12 \033[0m '
	echo '		--> Removing libxml2 temp install directory ...'
	
	rm -fr $installKitPath/libxml2-2.6.0
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when removing $installKitPath/libxml2-2.6.0"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $REMOVE_TEMP_DIRECTORY_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 12 OK \033[0m'
		echo
	fi 

	########## STEP 13: removing temp directory for mpich ##########
	echo -e  '	\033[40m\033[1;34m# STEP 13 \033[0m '
	echo '		--> Removing mpich2 temp install directory ...'
	
	rm -fr $installKitPath/mpich2-1.0.3
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when removing $installKitPath/mpich2-1.0.3"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $REMOVE_TEMP_DIRECTORY_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 13 OK \033[0m'
		echo
	fi 

	########## STEP 14: Configuring environment variables ##########
	echo -e  '	\033[40m\033[1;34m# STEP 14 \033[0m '
	echo '		--> Configuring environment variables for libxml2 and mpich2 ...'
	
	export LD_LIBRARY_PATH=$installKitPath/libxml2/lib:$LD_LIBRARY_PATH
	idx=$?	 
	export PATH=$installKitPath/libxml2/bin:$installKitPath/mpich2/bin:$PATH
	idx=`expr $idx + $?`
	echo "export LD_LIBRARY_PATH=$installKitPath/libxml2/lib:$LD_LIBRARY_PATH" >> $bashrcPath/.bashrc
	idx=`expr $idx + $?`
	echo "export PATH=$installKitPath/libxml2/bin:$installKitPath/mpich2/bin:$PATH" >> $bashrcPath/.bashrc
	idx=`expr $idx + $?`
	source $bashrcPath/.bashrc
	idx=`expr $idx + $?`
	if [ ! $(($idx)) = 0 ]
	then
		echo ''
		echo "	Error when configuring environment variables for libxml2 and mpich2"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $VAR_CONFIG_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 14 OK \033[0m'
		echo
	fi 

	
	######## STEP 15: installing paradiseo-peo ##########
	echo -e  '	\033[40m\033[1;34m# STEP 15 \033[0m '
	echo '		--> Installing Paradiseo-PEO ...'
	
	cd $installKitPath/paradiseo-peo && ./configure --with-EOdir=$installKitPath/paradiseo-eo/ --with-MOdir=$installKitPath/paradiseo-mo/ --with-MOEOdir=$installKitPath/paradiseo-moeo/ && make
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when installing Paradiseo-PEO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $PARADISEO_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 15 OK \033[0m'
		echo
	fi 

	
	######## STEP 16: copy .mpd.conf file to bashrc (required for ParadisEO) ##########
	echo -e  '	\033[40m\033[1;34m# STEP 16 \033[0m '
	echo '		--> Copy .mpd.conf file to bashrc path (required for Paradiseo-PEO)  ...'
	
	cp $resourceKitPath/.mpd.conf $bashrcPath/ && chmod 600 $bashrcPath/.mpd.conf
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "	Error when copying .mpd.conf file to bashrc path"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MPD_COPY_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 16 OK \033[0m'
		echo
	fi 	

	echo -e "\033[40m\033[1;33m### Now please run \"source $bashrcPath/.bashrc\" to save context ### \033[0m"
	sleep 2
	echo
	echo
	echo '=> ParadisEO install successful.To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr'
	echo
	return $SUCCESSFUL_PARADISEO_INSTALL
}


#------------------------------------------------------#
#-- FUNCTION   :  on_error			    ---#
#------------------------------------------------------#
#-- PARAMETERS :  				    ---#
#--	Error number	                            ---#
#------------------------------------------------------#
#-- RETURN:			                    ---#
#------------------------------------------------------#
function on_error()
{
	case $1 in
 	  $EO_UNPACKING_ERROR) 
		echo
		echo "  An error has occured : impossible to unpack paradiseo-eo archive " 
	  	echo "  Make sure that eo archive exists in current directory "
		echo 
		echo " => To report any problem of for help, please contact paradiseo-help@lists.gforge.inria.fr"
		echo ;;

  	  $MO_UNPACKING_ERROR) 
		echo
		echo "  An error has occured : impossible to unpack paradiseo-mo archive " 
	  	echo "  Make sure that mo archive exists in current directory "
		echo 
		echo "  => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr"
		echo ;;

	  $MOEO_UNPACKING_ERROR) 
		echo
		echo "  An error has occured : impossible to unpack paradiseo-moeo archive " 
	  	echo "  Make sure that moeo archive exists in current directory "
		echo 
		echo "  => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr"
		echo ;;
	
 	  $PARADISEO_UNPACKING_ERROR) 
		echo
		echo "  An error has occured : impossible to unpack paradiseo-peo archive " 
	  	echo "  Make sure that paradiseo archive exists in current directory "
		echo 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr"
		echo ;;

	$LIBXML_UNPACKING_ERROR) 
		echo
		echo "  An error has occured : impossible to unpack libxml2 archive" 
	  	echo "  Make sure that libxml2 archive exists in current directory"
		echo 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr"
		echo ;;

	$MPICH_UNPACKING_ERROR) 
		echo
		echo "  An error has occured : impossible to unpack mpich2 archive" 
	  	echo "  Make sure that mpich2 archive exists in current directory"
		echo 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr"
		echo ;;

	$EO_INSTALL_ERROR) 
		echo
		echo "  An error has occured : impossible to install Paradiseo-EO" 
	  	echo "If you need help, please contact paradiseo-help@lists.gforge.inria.fr"
		echo 
		echo ;;

	$MO_INSTALL_ERROR) 
		echo
		echo "  An error has occured : impossible to install Paradiseo-MO" 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr"
		echo ;;

	$MOEO_INSTALL_ERROR) 
		echo
		echo "  An error has occured : impossible to install Paradiseo-MOEO" 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr"
		echo ;;

	$PARADISEO_INSTALL_ERROR) 
		echo
		echo '  An error has occured : impossible to install Paradiseo-PEO' 
		echo '  Make sure you have the required variables in your environment (ex: by using "echo $PATH" for PATH variable) : '
		echo '	-LD_LIBRARY_PATH=<libxml2 install path>/libxml2/lib:$LD_LIBRARY_PATH'
		echo '	-PATH=<libxml2 install path>/libxml2/bin:<mpich2 install path>/mpich2/bin:$PATH'
		echo
		echo ' => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr'
		echo ;;

	*)
		echo
		echo ' => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr'
		echo ;;
 	 esac
}

#------------------------------------------------------#
#-- BODY   :  					    ---#
#------------------------------------------------------#

if [ "$1" = "--help" ]
then
	echo
	echo 'Use : ./install <install directory>  if your .bashrc file is in your home directory.'
	echo 'Example: ./install /home/work'
	echo
	echo 'Use : ./install <install directory> <bashrc path> to give your .bashrc path'
	echo 'Example: ./install /home/work /usr/mybashrcpath'
	echo
	echo '=> For further help, please contact paradiseo-help@lists.gforge.inria.fr'
	echo
	exit
fi

if [ "$1" = "" ]
then
	echo
	echo 'Error : please give a full path for target install'
	echo 'Use : ./install <install directory>'
	echo
else
	# check if the path is a full path and if it is valid
	cd / 
	cd $1 2>> /dev/null
	isFullPath=$?
	echo "$1" | grep "\."
	isRelativePath=$?

	if [ ! -d $1 ] || [ ! "$isFullPath" = "0" ] || [ "$isRelativePath" = "0" ]
	then
		echo
		echo "Error: Unknown directory $1 - Please give a valid directory for target install (full path)."
		echo
	else	
		if [ ! -f $HOME/.bashrc ]
		then
			if [ "$2" == "" ] || [ ! -f $2/.bashrc ]
			then
				echo " Please give a valid path for your bashrc file (use ./install --help for further information)"
			else
				bashrcPath=$2
				paradiseoInstall=run_install $1
				on_error $paradiseoInstall
			fi
		else
			bashrcPath=$HOME
			run_install $1
			paradiseoInstall=$?
			on_error $paradiseoInstall
		fi
	fi
fi



