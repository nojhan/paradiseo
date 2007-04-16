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
DIE=0
PROG=ParadisEO

# install types to select in the main menu
P_FULL_INSTALL=1
P_BASIC_INSTALL=2
P_PARALLEL_INSTALL=3
P_RM_PREVIOUS_INSTALLL=4
P_EXIT_INSTALL=5


# install steps
S_INTRODUCTION=1000
S_UNPACK_EO=1001
S_UNPACK_LIBXML=1002
S_UNPACK_MPICH=1003
S_INSTALL_EO=1004
S_INSTALL_MO=1005
S_INSTALL_MOEO=1006
S_INSTALL_LIBXML=1007
S_INSTALL_MPICH=1008
S_INSTALL_PEO=1009
S_REMOVE_TEMP_LIBXML=1010
S_REMOVE_TEMP_MPICH=1011
S_CONFIGURE_ENV=1012
S_CONFIGURE_MPD=1013
S_PEO_CHECK=1014
S_REMOVE_INSTALL=1015
S_END=1016

# define what are the possible install and their content
FULL_INSTALL="$S_CONFIGURE_ENV $S_INTRODUCTION $S_UNPACK_EO $S_UNPACK_LIBXML $S_UNPACK_MPICH $S_INSTALL_EO $S_INSTALL_MO $S_INSTALL_MOEO $S_INSTALL_LIBXML $S_REMOVE_TEMP_LIBXML $S_INSTALL_MPICH $S_REMOVE_TEMP_MPICH $S_CONFIGURE_ENV $S_INSTALL_PEO  $S_CONFIGURE_MPD $S_END"

BASIC_INSTALL="$S_INTRODUCTION $S_UNPACK_EO $S_INSTALL_EO $S_INSTALL_MO $S_INSTALL_MOEO $S_END"

PARALLEL_INSTALL="$S_PEO_CHECK $S_INTRODUCTION $S_UNPACK_LIBXML $S_INSTALL_LIBXML $S_REMOVE_TEMP_LIBXML $S_UNPACK_MPICH $S_INSTALL_MPICH $S_REMOVE_TEMP_MPICH $S_CONFIGURE_ENV $S_INSTALL_PEO $S_CONFIGURE_MPD $S_END"

RM_PREVIOUS_INSTALL="$S_REMOVE_INSTALL"

#others
PARADISEO_EO_ARCHIVE=paradiseo-eo.tar.gz
LIBS_PATH=lib
LIBXML2_ARCHIVE=libxml2-2.6.0.tar.bz2
MPICH2_ARCHIVE=mpich2-1.0.3.tar.gz

# errors
SUCCESSFUL_STEP=0
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
LIBXML_INSTALL_ERROR=116
MPICH_INSTALL_ERROR=117
PEO_CHECK_ERROR=118
RM_PARADISEO_EO_ERROR=119
RM_UTIL_ERROR=120

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
		echo 
		kill $$
		;;

	$LIBXML_UNPACKING_ERROR) 
		echo
		echo "  An error has occured : impossible to unpack libxml2 archive.See $SPY for more details" 
	  	echo "  Make sure that libxml2 archive exists in current directory"
		echo 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr  and join $SPY"
		echo 
		kill $$;;

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
		echo 
		kill $$;;

	$MO_INSTALL_ERROR) 
		echo
		echo "  An error has occured : impossible to install Paradiseo-MO.See $SPY for more details" 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo 
		kill $$;;

	$MOEO_INSTALL_ERROR) 
		echo
		echo "  An error has occured : impossible to install Paradiseo-MOEO.See $SPY for more details" 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo 
		kill $$;;

	$PARADISEO_INSTALL_ERROR) 
		echo
		echo "  An error has occured : impossible to install Paradiseo-PEO.See $SPY for more details" 
		echo '  Make sure you have the required variables in your environment (ex: by using "echo $PATH" for PATH variable) : '
		echo '	-LD_LIBRARY_PATH=<libxml2 install path>/libxml2/lib:$LD_LIBRARY_PATH'
		echo '	-PATH=<libxml2 install path>/libxml2/bin:<mpich2 install path>/mpich2/bin:$PATH'
		echo
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo 
		kill $$;;

	$LIBXML_INSTALL_ERROR)
		echo
		echo "  An error has occured : impossible to install libxml2. See $SPY for more details" 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo 
		kill $$;;

	$MPICH_INSTALL_ERROR)
		echo
		echo "  An error has occured : impossible to install mpich2 See $SPY for more details" 
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo 
		kill $$;;

	$PEO_CHECK_ERROR)
		echo
		echo " If you want to install ParadisEO-PEO, you should remove the old directories of libxml2 or mpich2 or choose another location." 
		echo 
		kill $$;;

	$RM_PARADISEO_EO_ERROR)
		echo
		echo "  An error has occured : impossible to remove ParadisEO-EO. See $SPY for more details" 
		echo " You may not have a previous ParadisEO install available in the current directory"
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo 
		kill $$;;

	$RM_UTIL_ERROR)
		echo
		echo "  An error has occured : impossible to remove the previous install of mpich2 and libxml2. See $SPY for more details" 
		echo " You may not have a previous ParadisEO install available in the current directory"
		echo " => To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo 
		kill $$;;

	$SUCCESSFUL_STEP)
		;;
	*)
		echo 
		;;
 	 esac
}

#------------------------------------------------------#
#-- FUNCTION   :  run_install_step		    ---#
#------------------------------------------------------#
#-- PARAMETERS :  				    ---#
#-- 	install path	                            ---#
#-- 	step to launch (0,1 ...)                    ---#
#-- 	counter for loop                            ---#
#-- Major function for install                      ---#
#------------------------------------------------------#
#-- RETURN: 0 if install OK                         ---#
#------------------------------------------------------#

function run_install_step()
{
	installKitPath=$1
	stepToRun=$2        
	currentStepCounter=$3

	RETURN=0
	
	case "$stepToRun" in
	$S_INTRODUCTION)
		########## Introduction #########
		clear
		echo ""
		echo -e ' \033[40m\033[1;33m### ParadisEO install starting .... ### \033[0m '
		echo
		echo "Installing the environment for Paradiseo...Note that the librairies \"libxml2\" ans \"mpich2\" required for ParadisEO are provided with this package."
		sleep 3
	
		echo
		echo
		return $SUCCESSFUL_STEP
		;;

	$S_UNPACK_EO)
		##########  unpacking paradiseo-eo ##########
		echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Unpacking Paradiseo-EO (Evolving Objects) ...'
	
		execute_cmd "tar xvzf $resourceKitPath/$LIBS_PATH/$PARADISEO_EO_ARCHIVE --directory $installKitPath" "[$currentStepCounter] Unpack Paradiseo-EO" $SPY
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "		--> Error when unpacking Paradiseo-EO"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $EO_UNPACKING_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi
		;;

	$S_UNPACK_LIBXML)
		########## unpacking libxml2 ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Unpacking libxml2 (required for ParadisEO) ...'
		
		execute_cmd "tar xvjf $resourceKitPath/$LIBS_PATH/$LIBXML2_ARCHIVE  --directory $installKitPath" "[$currentStepCounter] Unpack Libxml2" $SPY
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "		--> Error when unpacking libxml2"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $LIBXML_UNPACKING_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;

	$S_UNPACK_MPICH)
		########## unpacking mpich2 ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Unpacking mpich2 (required for ParadisEO) ...'
		
		execute_cmd "tar xzvf $resourceKitPath/$LIBS_PATH/$MPICH2_ARCHIVE --directory $installKitPath" "[3] Unpack Mpich2" $SPY
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "		--> Error when unpacking mpich2"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $MPICH_UNPACKING_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;

	$S_INSTALL_EO)
		########## installing paradiseo-eo ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Installing Paradiseo-EO. Please wait ...'
		
		execute_cmd "cd $installKitPath/paradiseo-eo" "[$currentStepCounter-1] Go in Paradiseo-EO dir"  $SPY 
		RETURN=`expr $RETURN + $?`
		execute_cmd "./autogen.sh" "[$currentStepCounter-2] Run autogen"  $SPY
		RETURN=`expr $RETURN + $?`
		execute_cmd "./configure" "[$currentStepCounter-3] Run configure"  $SPY
		RETURN=`expr $RETURN + $?`
		execute_cmd "make" "[$currentStepCounter-4] Compile ParadisEO-EO"  $SPY
		RETURN=`expr $RETURN + $?`
		if [ ! $(($RETURN)) = 0 ]
		then
			echo ''
			echo "		--> Error when installing Paradiseo-EO"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $EO_INSTALL_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo 
			return $SUCCESSFUL_STEP
		fi 
		;;
	$S_INSTALL_MO)
		##########  installing paradiseo-mo ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Installing Paradiseo-MO. Please wait ...'
	
		execute_cmd "cd $installKitPath/paradiseo-mo" "[$currentStepCounter-1] Go in Paradiseo-MO dir"  $SPY 
		RETURN=$?
		execute_cmd "./autogen.sh --with-EOdir=$installKitPath/paradiseo-eo" "[$currentStepCounter-2] Run autogen"  $SPY
		RETURN=`expr $RETURN + $?`
		execute_cmd "make" "[$currentStepCounter-3] Compile ParadisEO-MO"  $SPY
		RETURN=`expr $RETURN + $?`
		if [ ! $(($RETURN)) = 0 ]
		then
			echo ''
			echo "		--> Error when installing Paradiseo-MO"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $MO_INSTALL_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;
	$S_INSTALL_MOEO)
		########## installing MOEO ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Installing Paradiseo-MOEO. Please wait ...'
		
		execute_cmd "cd $installKitPath/paradiseo-moeo" "[$currentStepCounter-1] Go in Paradiseo-MOEO dir"  $SPY 
		RETURN=$?
		execute_cmd "./autogen.sh --with-EOdir=$installKitPath/paradiseo-eo" "[$currentStepCounter-2] Run autogen"  $SPY
		RETURN=`expr $RETURN + $?`
		execute_cmd "make" "[$currentStepCounter-3] Compile ParadisEO-MOEO"  $SPY
		RETURN=`expr $RETURN + $?`
		if [ ! $(($RETURN)) = 0 ]
		then
			echo ''
			echo "		--> Error when installing Paradiseo-MOEO"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $MOEO_INSTALL_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;
	$S_INSTALL_LIBXML)
		########## installing LIBXML2 ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Installing libxml2. Please wait ...'
		
		execute_cmd "mkdir $installKitPath/libxml2" "[$currentStepCounter-1] Create libxml2 dir"  $SPY 
		RETURN=$?
		execute_cmd "cd $installKitPath/libxml2-2.6.0/" "[$currentStepCounter-2] Go in libxml2-2.6.0 dir"  $SPY
		RETURN=`expr $RETURN + $?`
		execute_cmd "./configure --prefix=$installKitPath/libxml2/ --exec-prefix=$installKitPath/libxml2/" "[$currentStepCounter-3] Run configure for libxml2"  $SPY
		RETURN=`expr $RETURN + $?`
		execute_cmd "make" "[$currentStepCounter-4] Compile libxml2"  $SPY
		RETURN=`expr $RETURN + $?`
		execute_cmd "make install" "[$currentStepCounter-5] Run install libxml2 "  $SPY
		RETURN=`expr $RETURN + $?`
		if [ ! $(($RETURN)) = 0 ]
		then
			echo ''
			echo "		--> Error when installing libxml2"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $LIBXML_INSTALL_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;
	$S_INSTALL_MPICH)
		########## installing MPICH2 ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Installing mpich2. Please wait ...'
		
		execute_cmd "mkdir $installKitPath/mpich2" "[$currentStepCounter-1] Create mpich2 dir"  $SPY 
		RETURN=$?
		execute_cmd "cd $installKitPath/mpich2-1.0.3/" "[$currentStepCounter-2] Go in mpich2-1.0.3 dir"  $SPY
		RETURN=`expr $RETURN + $?`
		execute_cmd "./configure --prefix=$installKitPath/mpich2/" "[$currentStepCounter-3] Run configure for mpich2"  $SPY
		RETURN=`expr $RETURN + $?`
		execute_cmd "make" "[$currentStepCounter-4] Compile mpich2"  $SPY
		RETURN=`expr $RETURN + $?`
		execute_cmd "make install" "[$currentStepCounter-5] Run install mpich2 "  $SPY
		RETURN=`expr $RETURN + $?`
		if [ ! $(($RETURN)) = 0 ]
		then
			echo ''
			echo "		--> Error when installing MPICH2"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $MPICH_INSTALL_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;
	$S_REMOVE_TEMP_LIBXML)
		########## removing temp directory for libxml ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Removing libxml2 temp install directory ...'
		
		execute_cmd "rm -fr $installKitPath/libxml2-2.6.0" "[$currentStepCounter] Remove Libxml2 temporary directory" $SPY
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "		--> Error when removing $installKitPath/libxml2-2.6.0"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $REMOVE_TEMP_DIRECTORY_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;

	$S_REMOVE_TEMP_MPICH)
		########## removing temp directory for mpich ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Removing mpich2 temp install directory ...'
		
		execute_cmd "rm -fr $installKitPath/mpich2-1.0.3" "[$currentStepCounter] Remove Mpich2 temporary directory" $SPY
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "		--> Error when removing $installKitPath/mpich2-1.0.3"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $REMOVE_TEMP_DIRECTORY_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;

	$S_REMOVE_INSTALL)
		########## removing a previous install of EO ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Removing your previous install of ParadisEO ...'
	
		execute_cmd "rm -r $installKitPath/paradiseo-eo " "[$currentStepCounter] Remove previous version of ParadisEO-EO" $SPY
		if [ ! "$?" = "0" ]
		then
			echo ''
			echo "		--> Error when removing $installKitPath/paradiseo-eo"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $RM_PARADISEO_EO_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 

		if [ -d "$installKitPath/mpich2" ]
		then
			execute_cmd "rm -r $installKitPath/mpich2" "[$currentStepCounter] Remove previous install of mpich2" $SPY 
		fi
		idx=$?

		if [ -d "$installKitPath/libxml2" ]
		then
			execute_cmd "rm -r $installKitPath/libxml2" "[$currentStepCounter] Remove previous install of libxml2" $SPY 
		fi
		idx=`expr $idx + $?`

		if [ ! $(($idx)) = 0 ]
		then
			echo ''
			echo "		--> Error when removing previous install of libxml2 and mpich2"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $RM_UTIL_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;

	$S_CONFIGURE_ENV)
		########## Configuring environment variables ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Configuring environment variables for libxml2 and mpich2 ...'
		
		execute_cmd "export LD_LIBRARY_PATH=$installKitPath/libxml2/lib:\$LD_LIBRARY_PATH" "[$currentStepCounter-1] Export LD_LIBRARY_PATH variable" $SPY
		idx=$?	 
		execute_cmd "export PATH=$installKitPath/libxml2/bin:$installKitPath/mpich2/bin:\$PATH" "[$currentStepCounter-2] Export PATH variable" $SPY 
	
		execute_cmd "echo export LD_LIBRARY_PATH=$installKitPath/libxml2/lib:\$LD_LIBRARY_PATH" "[$currentStepCounter-3] Export LD_LIBRARY_PATH variable into env" $SPY $homePath/.bashrc
		idx=$?	 

		execute_cmd "echo export PATH=$installKitPath/libxml2/bin:$installKitPath/mpich2/bin:\$PATH" "[$currentStepCounter-4] Export PATH variable into env" $SPY $homePath/.bashrc
		idx=`expr $idx + $?`

		execute_cmd "source $homePath/.bashrc" "[$currentStepCounter-5] Export variables" $SPY
		idx=`expr $idx + $?`

		if [ ! $(($idx)) = 0 ]
		then
			echo ''
			echo "		--> Error when configuring environment variables for libxml2 and mpich2"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $VAR_CONFIG_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;
	$S_INSTALL_PEO)
		######## installing paradiseo-peo ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Installing Paradiseo-PEO. Please wait ...'
		
		execute_cmd "cd $installKitPath/paradiseo-peo" "[$currentStepCounter-1] Go in Paradiseo-PEO dir"  $SPY 
		RETURN=$?
		execute_cmd " ./autogen.sh --with-EOdir=$installKitPath/paradiseo-eo/ --with-MOdir=$installKitPath/paradiseo-mo/ --with-MOEOdir=$installKitPath/paradiseo-moeo" "[$currentStepCounter-2] Run autogen for ParadisEO-PEO"  $SPY
		RETURN=`expr $RETURN + $?`
		execute_cmd "make" "[$currentStepCounter-3] Compile ParadisEO-PEO "  $SPY
		RETURN=`expr $RETURN + $?`
		if [ ! $(($RETURN)) = 0 ]
		then
			echo ''
			echo "		--> Error when installing Paradiseo-PEO"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $PARADISEO_INSTALL_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;
	$S_CONFIGURE_MPD)
		######## copy .mpd.conf file in your HOME directory or in /etc if you are root (required for mpich2) 
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Copy .mpd.conf file in your HOME directory or in /etc if you are root (required for mpich2)  ...'
		if [ "$UID" = "0" ]
		then
			execute_cmd "cp $resourceKitPath/.mpd.conf /etc" "[$currentStepCounter-1] Copy mpd.conf file in /etc (root)" $SPY
			RETURN=$?
			execute_cmd "mv /etc/.mpd.conf /etc/mpd.conf" "[$currentStepCounter-2] Move .mpd.conf to mpd.conf" $SPY
			RETURN=`expr $RETURN + $?`
			execute_cmd "chmod 600 /etc/mpd.conf" "[$currentStepCounter-3] Change .mpd.conf rights" $SPY
			RETURN=`expr $RETURN + $?`
		else
			execute_cmd "cp $resourceKitPath/.mpd.conf $HOME" "[$currentStepCounter-1] Copy mpd.conf file in in your HOME directory" $SPY
			RETURN=$?
			execute_cmd "chmod 600 $HOME/.mpd.conf" "[$currentStepCounter-2] Change .mpd.conf rights" $SPY
			RETURN=`expr $RETURN + $?`
		fi
		if [ ! $(($RETURN)) = 0 ]
		then
			echo ''
			echo "		--> Error when copying .mpd.conf file "
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $MPD_COPY_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 	
		;;

	$S_PEO_CHECK)
		if ( [ -d paradiseo-eo ] && [ ! -d paradiseo-mo ] && [ ! -d paradiseo-moeo ] )
		then
			if ( [ -d libxml2 ] || [ -d mpich2 ] )
			then
				echo 
				echo "A previous installation of ParadisEO-PEO may exist because libxml2 or mpich2 directory have been detected in $installKitPath."
				echo -e " \033[40m\033[1;33m	=> Do you want to remove these directories for a new installation ? If you choose NO, the installation will stop. (y/n) ? \033[0m "
				read ANSWER
				if [ "$ANSWER" = "y" ]
				then
					execute_cmd "rm -rf $installKitPath/libxml2 $installKitPath/mpich2" "[$currentStepCounter] Remove libxml2 ans mpich2 directories for a new install" $SPY "/dev/null" "/dev/null"
				else
					return $PEO_CHECK_ERROR
				fi
			fi 
		else			
			echo 
			echo "Basic install not found (at least one of the EO,MO,MOEO components is missing) in $installKitPath."
			
			execute_cmd "test -d paradiseo-eo" "[$currentStepCounter-1] Check previous basic install" $SPY
			execute_cmd "test -d paradiseo-mo" "[$currentStepCounter-2] Check previous basic install" $SPY
			execute_cmd "test -d paradiseo-moeo" "[$currentStepCounter-3] Check previous basic install" $SPY

			echo ''
			echo "		--> Error when searching for a previous basic install in $installKitPath."
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $RM_PARADISEO_EO_ERROR	
		fi
		;;
	$S_END)
		echo -e "\033[40m\033[1;33m### Now please run \"source $homePath/.bashrc\" to save the context ### \033[0m"
		sleep 2
		echo
		echo
		echo '=> ParadisEO install successful.To report any problem or for help, please contact paradiseo-help@lists.gforge.inria.fr'
		echo
		return $SUCCESSFUL_STEP
		;;
	*)
		
	;;
	esac
}


#------------------------------------------------------#
#-- BODY   :  					    ---#
#------------------------------------------------------#

#check if we have all we need
(autoconf --version) < /dev/null > /dev/null 2>&1 ||
{
    echo "You must have autoconf installed to compile $PROG. Please update your system to get it before installing $PROG."
    execute_cmd "echo \"You must have autoconf installed to compile $PROG. Please update your system to get it before installing $PROG.\"" "[0-1] Check autoconf" $SPY    
    DIE=1
}

(automake --version) < /dev/null > /dev/null 2>&1 ||
{
    echo "You must have automake installed to compile $PROG. Please update your system to get it before installing $PROG."
    execute_cmd "echo \"You must have automake installed to compile $PROG. Please update your system to get it before installing $PROG.\"" "[0-2] Check autoconf" $SPY 
    DIE=1
}
if [ "$DIE" = "1" ] 
then
    exit 1
fi


# main
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
	if [ "$1" = "" ]
	then
		echo " Please give a valid path for your home directory (use ./installParadiseo.sh --help for further information)"
	else
		homePath=$1
	fi
else
	homePath=$HOME
fi


# simple menu
INSTALL_TREATENED=0
INSTALL_PATH=$PWD

while [ ! "$INSTALL_TREATENED" = "1" ]
do	
	case "$INSTALL_TYPE" in
	$P_FULL_INSTALL)
		counter=0
		for step in $FULL_INSTALL	
		do
			run_install_step $INSTALL_PATH $step $counter
			on_error $?
			counter=`expr $counter + 1`
		done
		INSTALL_TREATENED=1
		;;

	$P_BASIC_INSTALL)
		counter=0
		for step in $BASIC_INSTALL	
		do
			run_install_step $INSTALL_PATH $step $counter
			on_error $?
			counter=`expr $counter + 1`
		done
		INSTALL_TREATENED=1
		;;

	$P_PARALLEL_INSTALL)
		counter=0
		for step in $PARALLEL_INSTALL	
		do
			run_install_step $INSTALL_PATH $step $counter
			on_error $?
			counter=`expr $counter + 1`
		done
		INSTALL_TREATENED=1
		;;

	$P_RM_PREVIOUS_INSTALLL)
		counter=0
		for step in $RM_PREVIOUS_INSTALL	
		do
			run_install_step $INSTALL_PATH $step $counter
			on_error $?
			counter=`expr $counter + 1`
		done
		INSTALL_TREATENED=1
		;;

	$P_EXIT_INSTALL)
		INSTALL_TREATENED=1
		;;

	*)
		echo
		echo -e ' \033[40m\033[1;33m### Please select your install for ParadisEO : ### \033[0m '
		echo
		echo "	 1 : Full install (all the components : EO,MO,MOEO and PEO)"
		echo "	 2 : Basic install: only EO,MO and MOEO components will be installed."
		echo "	 3 : ParadisEO-PEO install. I've already installed the basic version and I want to install ParadisEO-PEO"
		echo "	 4 : Remove a previous install of ParadisEO located in $INSTALL_PATH"
		echo "	 5 : Exit install"
		read INSTALL_TYPE
	;;
	esac
done




