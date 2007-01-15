#!/bin/sh

#########################################################################################
#
#	Project		:	paradisEO full package install
#	File		:	installParadiseo.sh
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
LIBS_PATH=lib
LIBXML2_ARCHIVE=libxml2-2.6.0.tar.bz2
MPICH2_ARCHIVE=mpich2-1.0.3.tar.gz

# errors
SUCCESSFUL_PARADISEO_INSTALL=0
EO_UNPACKING_ERROR=100
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

#Date
DATE=`/bin/date '+%Y%m%d%H%M%S'`
# create log file
SPY=$PWD/install-paradiseo.${DATE}.log

#------------------------------------------------------#
#-- FUNCTION   :  execute_cmd		            ---#
#------------------------------------------------------#
#-- PARAMETERS :  				    ---#
#--	$1 : cmd line                               ---#
#--	$2 : comment                                ---#
#--	$3 : spy file                               ---#
#--	$4 : output std file                        ---#
#--	$5 : error log file                         ---#
#--                                                 ---#
#------------------------------------------------------#
#-- CODE RETURN : 0 : OK                            ---#
#-- CODE RETURN : 1 : NOK                           ---#
#------------------------------------------------------#
function execute_cmd
{
	COMMAND=${1}
	COMMENT=${2}
	FIC_ESP=${3}
	FIC_OUT=${4}
	FIC_ERR=${5}

	if [ `echo ${FIC_OUT} | wc -c` -eq 1 ]
	then
		FIC_OUT=${FIC_ESP}
	fi

	if [ `echo ${FIC_ERR} | wc -c` -eq 1 ]
	then
		FIC_ERR=${FIC_ESP}
	fi

	echo "" >> ${FIC_ESP}
	echo "[execute_cmd][Begin][`/bin/date +%H:%M:%S`]" >> ${FIC_ESP}

	echo "------------------------------------------------------------------------------------------------------------" >> ${FIC_ESP}
	echo "${COMMENT}" >> ${FIC_ESP}
	echo "------------------------------------------------------------------------------------------------------------" >> ${FIC_ESP}
	echo "${COMMAND}" >> ${FIC_ESP}
	
	${COMMAND} >> ${FIC_OUT} 2>> ${FIC_ERR}
	
	RETURN_CODE=$?
	echo "RETURN_CODE : ${RETURN_CODE}" >> ${FIC_ESP}
	
	if [ ${RETURN_CODE} -eq 0 ]
	then
		echo "     ${COMMENT} OK" >> ${FIC_ESP}
		echo "[execute_cmd][End][`/bin/date +%H:%M:%S`]" >> ${FIC_ESP}
		return 0
	else
		echo "     $ERROR_TAG ${COMMENT} NOK" >> ${FIC_ESP}
		return 1
	fi
}

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
	RETURN=0

	########## STEP 0 : introduction ##########
	clear
	echo ""
	echo -e ' \033[40m\033[1;33m### ParadisEO install starting .... ### \033[0m '
	#sleep 4
	echo
	echo "Installing the environment for Paradiseo... this may take about 15 minutes to complete. Note that the librairies \"libxml2\" ans \"mpich2\" required for ParadisEO are provided with this package."
	sleep 3

	echo
	echo

	########## STEP 1: unpacking paradiseo-eo ##########
	echo -e '	\033[40m\033[1;34m# STEP 1 \033[0m '
	echo '		--> Unpacking Paradiseo-EO (Evolving Objects) ...'

	execute_cmd "tar xvf $resourceKitPath/$LIBS_PATH/$PARADISEO_EO_ARCHIVE --directory $installKitPath" "[1] Unpack Paradiseo-EO" $SPY
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "		--> Error when unpacking Paradiseo-EO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $EO_UNPACKING_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 1 OK \033[0m'
		echo
	fi 
	

	########## STEP 2: unpacking libxml2 ##########
	echo -e  '	\033[40m\033[1;34m# STEP 2 \033[0m '
	echo '		--> Unpacking libxml2 (required for ParadisEO) ...'
	
	execute_cmd "tar xvjf $resourceKitPath/$LIBS_PATH/$LIBXML2_ARCHIVE  --directory $installKitPath" "[2] Unpack Libxml2" $SPY
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "		--> Error when unpacking libxml2"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $LIBXML_UNPACKING_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 2 OK \033[0m'
		echo
	fi 

	########## STEP 3: unpacking mpich2 ##########
	echo -e  '	\033[40m\033[1;34m# STEP 3 \033[0m '
	echo '		--> Unpacking mpich2 (required for ParadisEO) ...'
	
	execute_cmd "tar xzvf $resourceKitPath/$LIBS_PATH/$MPICH2_ARCHIVE --directory $installKitPath" "[3] Unpack Mpich2" $SPY
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "		--> Error when unpacking mpich2"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MPICH_UNPACKING_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 3 OK \033[0m'
		echo
	fi 

	########## STEP 4: installing paradiseo-eo ##########
	echo -e  '	\033[40m\033[1;34m# STEP 4 \033[0m '
	echo '		--> Installing Paradiseo-EO. Please wait ...'
	
	execute_cmd "cd $installKitPath/paradiseo-eo" "[4-1] Go in Paradiseo-EO dir"  $SPY 
	RETURN=`expr $RETURN + $?`
	execute_cmd "./autogen.sh" "[4-2] Run autogen"  $SPY
	RETURN=`expr $RETURN + $?`
	execute_cmd "./configure" "[4-3] Run configure"  $SPY
	RETURN=`expr $RETURN + $?`
	execute_cmd "make" "[4-4] Compile ParadisEO-EO"  $SPY
	RETURN=`expr $RETURN + $?`
	if [ ! $(($RETURN)) = 0 ]
	then
		echo ''
		echo "		--> Error when installing Paradiseo-EO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $EO_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 4 OK \033[0m'
		echo 
	fi 

	########## STEP 5: installing paradiseo-mo ##########
	echo -e  '	\033[40m\033[1;34m# STEP 5 \033[0m '
	echo '		--> Installing Paradiseo-MO. Please wait ...'

	execute_cmd "cd $installKitPath/paradiseo-mo" "[5-1] Go in Paradiseo-MO dir"  $SPY 
	RETURN=$?
	execute_cmd "./autogen.sh --with-EOdir=$installKitPath/paradiseo-eo" "[5-2] Run autogen"  $SPY
	RETURN=`expr $RETURN + $?`
	execute_cmd "make" "[5-3] Compile ParadisEO-MO"  $SPY
	RETURN=`expr $RETURN + $?`
	if [ ! $(($RETURN)) = 0 ]
	then
		echo ''
		echo "		--> Error when installing Paradiseo-MO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MO_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 5 OK \033[0m'
		echo
	fi 

	########## STEP 6: installing MOEO ##########
	echo -e  '	\033[40m\033[1;34m# STEP 6 \033[0m '
	echo '		--> Installing Paradiseo-MOEO. Please wait ...'
	
	execute_cmd "cd $installKitPath/paradiseo-moeo" "[6-1] Go in Paradiseo-MOEO dir"  $SPY 
	RETURN=$?
	execute_cmd "./autogen.sh --with-EOdir=$installKitPath/paradiseo-eo" "[6-2] Run autogen"  $SPY
	RETURN=`expr $RETURN + $?`
	execute_cmd "make" "[6-3] Compile ParadisEO-MOEO"  $SPY
	RETURN=`expr $RETURN + $?`
	if [ ! $(($RETURN)) = 0 ]
	then
		echo ''
		echo "		--> Error when installing Paradiseo-MOEO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MOEO_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 6 OK \033[0m'
		echo
	fi 

	########## STEP 7: installing LIBXML2 ##########
	echo -e  '	\033[40m\033[1;34m# STEP 7 \033[0m '
	echo '		--> Installing LIBXML2. Please wait ...'
	
	execute_cmd "mkdir $installKitPath/libxml2" "[7-1] Create libxml2 dir"  $SPY 
	RETURN=$?
	execute_cmd "cd $installKitPath/libxml2-2.6.0/" "[7-2] Go in libxml2-2.6.0 dir"  $SPY
	RETURN=`expr $RETURN + $?`
	execute_cmd "./configure --prefix=$installKitPath/libxml2/ --exec-prefix=$installKitPath/libxml2/" "[7-3] Run configure for libxml2"  $SPY
	RETURN=`expr $RETURN + $?`
	execute_cmd "make" "[7-4] Compile libxml2"  $SPY
	RETURN=`expr $RETURN + $?`
	execute_cmd "make install" "[7-5] Run install libxml2 "  $SPY
	RETURN=`expr $RETURN + $?`
	if [ ! $(($RETURN)) = 0 ]
	then
		echo ''
		echo "		--> Error when installing LIBXML2"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $LIBXML_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 7 OK \033[0m'
		echo
	fi 


	########## STEP 8: installing MPICH2 ##########
	echo -e  '	\033[40m\033[1;34m# STEP 8 \033[0m '
	echo '		--> Installing MPICH2. Please wait ...'
	
	execute_cmd "mkdir $installKitPath/mpich2" "[8-1] Create mpich2 dir"  $SPY 
	RETURN=$?
	execute_cmd "cd $installKitPath/mpich2-1.0.3/" "[8-2] Go in mpich2-1.0.3 dir"  $SPY
	RETURN=`expr $RETURN + $?`
	execute_cmd "./configure --prefix=$installKitPath/mpich2/" "[8-3] Run configure for mpich2"  $SPY
	RETURN=`expr $RETURN + $?`
	execute_cmd "make" "[8-4] Compile mpich2"  $SPY
	RETURN=`expr $RETURN + $?`
	execute_cmd "make install" "[8-5] Run install mpich2 "  $SPY
	RETURN=`expr $RETURN + $?`
	if [ ! $(($RETURN)) = 0 ]
	then
		echo ''
		echo "		--> Error when installing MPICH2"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MPICH_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 8 OK \033[0m'
		echo
	fi 

	########## STEP 9: removing temp directory for libxml ##########
	echo -e  '	\033[40m\033[1;34m# STEP 9 \033[0m '
	echo '		--> Removing libxml2 temp install directory ...'
	
	execute_cmd "rm -fr $installKitPath/libxml2-2.6.0" "[9] Remove Libxml2 temporary directory" $SPY
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "		--> Error when removing $installKitPath/libxml2-2.6.0"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $REMOVE_TEMP_DIRECTORY_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 9 OK \033[0m'
		echo
	fi 

	########## STEP 10: removing temp directory for mpich ##########
	echo -e  '	\033[40m\033[1;34m# STEP 10 \033[0m '
	echo '		--> Removing mpich2 temp install directory ...'
	
	execute_cmd "rm -fr $installKitPath/mpich2-1.0.3" "[10] Remove Mpich2 temporary directory" $SPY
	if [ ! "$?" = "0" ]
	then
		echo ''
		echo "		--> Error when removing $installKitPath/mpich2-1.0.3"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $REMOVE_TEMP_DIRECTORY_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 10 OK \033[0m'
		echo
	fi 


	########## STEP 11: Configuring environment variables ##########
	echo -e  '	\033[40m\033[1;34m# STEP 11 \033[0m '
	echo '		--> Configuring environment variables for libxml2 and mpich2 ...'
	
	execute_cmd "export LD_LIBRARY_PATH=$installKitPath/libxml2/lib:$LD_LIBRARY_PATH" "[11-1] Export LD_LIBRARY_PATH variable" $SPY
	idx=$?	 
	execute_cmd "export PATH=$installKitPath/libxml2/bin:$installKitPath/mpich2/bin:$PATH" "[11-2] Export PATH variable" $SPY 

	execute_cmd "echo export LD_LIBRARY_PATH=$installKitPath/libxml2/lib:$LD_LIBRARY_PATH" "[11-3] Export LD_LIBRARY_PATH variable into env" $SPY $homePath/.bashrc
	idx=$?	 
	execute_cmd "echo export PATH=$installKitPath/libxml2/bin:$installKitPath/mpich2/bin:$PATH" "[11-4] Export PATH variable into env" $SPY $homePath/.bashrc
	idx=`expr $idx + $?`
	execute_cmd "source $homePath/.bashrc" "[11-3] Export variables" $SPY
	idx=`expr $idx + $?`
	if [ ! $(($idx)) = 0 ]
	then
		echo ''
		echo "		--> Error when configuring environment variables for libxml2 and mpich2"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $VAR_CONFIG_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 11 OK \033[0m'
		echo
	fi 

	
	######## STEP 12: installing paradiseo-peo ##########
	echo -e  '	\033[40m\033[1;34m# STEP 12 \033[0m '
	echo '		--> Installing Paradiseo-PEO. Please wait ...'
	
	execute_cmd "cd $installKitPath/paradiseo-peo" "[12-1] Go in Paradiseo-PEO dir"  $SPY 
	RETURN=$?
	execute_cmd " ./autogen.sh --with-EOdir=$installKitPath/paradiseo-eo/ --with-MOdir=$installKitPath/paradiseo-mo/ --with-MOEOdir=$installKitPath/paradiseo-moeo" "[12-2] Run autogen for ParadisEO-PEO"  $SPY
	RETURN=`expr $RETURN + $?`
	execute_cmd "make" "[12-3] Compile ParadisEO-PEO "  $SPY
	RETURN=`expr $RETURN + $?`
	if [ ! $(($RETURN)) = 0 ]
	then
		echo ''
		echo "		--> Error when installing Paradiseo-PEO"
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $PARADISEO_INSTALL_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 12 OK \033[0m'
		echo
	fi 

	
	######## STEP 13: copy .mpd.conf file in your HOME directory or in /etc if you are root (required for mpich2) 
	echo -e  '	\033[40m\033[1;34m# STEP 13 \033[0m '
	echo '		--> Copy .mpd.conf file in your HOME directory or in /etc if you are root (required for mpich2)  ...'
	if [ "$UID" = "0" ]
	then
		execute_cmd "cp $resourceKitPath/.mpd.conf /etc" "[13-1] Copy mpd.conf file in /etc (root)" $SPY
		RETURN=$?
 		execute_cmd "mv /etc/.mpd.conf /etc/mpd.conf" "[13-2] Move .mpd.conf to mpd.conf" $SPY
		RETURN=`expr $RETURN + $?`
 		execute_cmd "chmod 600 /etc/mpd.conf" "[13-3] Change .mpd.conf rights" $SPY
		RETURN=`expr $RETURN + $?`
	else
		execute_cmd "cp $resourceKitPath/.mpd.conf $HOME" "[13-1] Copy mpd.conf file in in your HOME directory" $SPY
		RETURN=$?
 		execute_cmd "chmod 600 $HOME/.mpd.conf" "[13-2] Change .mpd.conf rights" $SPY
		RETURN=`expr $RETURN + $?`
	fi
	if [ ! $(($RETURN)) = 0 ]
	then
		echo ''
		echo "		--> Error when copying .mpd.conf file "
		echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		return $MPD_COPY_ERROR
	else
		echo -e '	\033[40m\033[1;34m# STEP 13 OK \033[0m'
		echo
	fi 	

	echo -e "\033[40m\033[1;33m### Now please run \"source $homePath/.bashrc\" to save context ### \033[0m"
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
		echo "  An error has occured : impossible to unpack paradiseo-eo archive.See $SPY for more details" 
	  	echo "  Make sure that eo archive exists in current directory "
		echo 
		echo " => To report any problem of for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo ;;

	$LIBXML_UNPACKING_ERROR) 
		echo
		echo "  An error has occured : impossible to unpack libxml2 archive.See $SPY for more details" 
	  	echo "  Make sure that libxml2 archive exists in current directory"
		echo 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr  and join $SPY"
		echo ;;

	$MPICH_UNPACKING_ERROR) 
		echo
		echo "  An error has occured : impossible to unpack mpich2 archive.See $SPY for more details" 
	  	echo "  Make sure that mpich2 archive exists in current directory"
		echo 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr  and join $SPY"
		echo ;;

	$EO_INSTALL_ERROR) 
		echo
		echo "  An error has occured : impossible to install Paradiseo-EO.See $SPY for more details" 
	  	echo "If you need help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo 
		echo ;;

	$MO_INSTALL_ERROR) 
		echo
		echo "  An error has occured : impossible to install Paradiseo-MO.See $SPY for more details" 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo ;;

	$MOEO_INSTALL_ERROR) 
		echo
		echo "  An error has occured : impossible to install Paradiseo-MOEO.See $SPY for more details" 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo ;;

	$PARADISEO_INSTALL_ERROR) 
		echo
		echo "  An error has occured : impossible to install Paradiseo-PEO.See $SPY for more details" 
		echo '  Make sure you have the required variables in your environment (ex: by using "echo $PATH" for PATH variable) : '
		echo '	-LD_LIBRARY_PATH=<libxml2 install path>/libxml2/lib:$LD_LIBRARY_PATH'
		echo '	-PATH=<libxml2 install path>/libxml2/bin:<mpich2 install path>/mpich2/bin:$PATH'
		echo
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo ;;

	*)
		echo
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo ;;
 	 esac
}

#------------------------------------------------------#
#-- BODY   :  					    ---#
#------------------------------------------------------#

if [ "$1" = "--help" ]
then
	echo
	echo 'Use : ./installParadiseo.sh for standard install'
	echo
	echo 'Use : ./installParadiseo.sh <HOME path> to give your HOME path'
	echo 'Example: ./installParadiseo.sh /usr/home/me'
	echo
	echo '=> For further help, please contact paradiseo-help@lists.gforge.inria.fr'
	echo
	exit
fi

if [ ! -d $HOME ]
then
	if [ "$1" == "" ]
	then
		echo " Please give a valid path for your home directory (use ./installParadiseo.sh --help for further information)"
	else
		homePath=$1
	fi
else
	homePath=$HOME
fi


# That's it !
run_install $PWD
paradiseoInstall=$? 
on_error $paradiseoInstall


