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
CMAKE_PRIMARY_CONFIG_FILE=install.cmake
HOME_PATH=$HOME

# generator types available on Unix platforms
P_UNIX_MAKEFILES=1
P_KDEVELOP3_PROJECT=2
P_XCODE_PROJECT=3
G_UNIX_MAKEFILES="Unix Makefiles"
G_KDEVELOP3_PROJECT="KDevelop3"
G_XCODE_PROJECT="Xcode"

# should we compile ParadisEO ?
COMPILE_PARADISEO=1

# Build types
DEFAULT_BUILD_TYPE=Release
BUILD_TYPE=$DEFAULT_BUILD_TYPE

# CMake/CTest/Dart flags
CTEST_DEFAULT_CONFIG="-D ExperimentalStart -D ExperimentalBuild"
CTEST_CONFIG=$CTEST_DEFAULT_CONFIG

# install types to select in the main menu
P_FULL_INSTALL=1
P_BASIC_INSTALL=2
P_PARALLEL_INSTALL=3
P_RM_PREVIOUS_INSTALLL=4
P_EXIT_INSTALL=5

IS_MPICH_INSTALLED=1
IS_LIBXML2_INSTALLED=1
USE_EXISTING_MPICH=-1
USE_EXISTING_LIBXML2=-1

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
S_END_WITHOUT_INFO=1017
S_CHECK_AUTOTOOLS=1018

#### define what are the possible installs and their content

# full install
FULL_INSTALL="$S_INTRODUCTION $S_UNPACK_LIBXML $S_UNPACK_MPICH $S_INSTALL_EO $S_INSTALL_MO $S_INSTALL_MOEO $S_CHECK_AUTOTOOLS $S_INSTALL_LIBXML $S_REMOVE_TEMP_LIBXML $S_INSTALL_MPICH $S_REMOVE_TEMP_MPICH $S_CONFIGURE_ENV $S_INSTALL_PEO  $S_CONFIGURE_MPD $S_END"

FULL_INSTALL_WITHOUT_LIBXML2="$S_INTRODUCTION $S_UNPACK_MPICH $S_INSTALL_EO $S_INSTALL_MO $S_INSTALL_MOEO $S_CHECK_AUTOTOOLS $S_INSTALL_MPICH $S_REMOVE_TEMP_MPICH $S_CONFIGURE_MPICH_ENV $S_INSTALL_PEO  $S_CONFIGURE_MPD $S_END"

FULL_INSTALL_WITHOUT_MPICH2="$S_INTRODUCTION $S_UNPACK_LIBXML $S_INSTALL_EO $S_INSTALL_MO $S_INSTALL_MOEO $S_CHECK_AUTOTOOLS $S_INSTALL_LIBXML $S_REMOVE_TEMP_LIBXML $S_CONFIGURE_LIBXML2_ENV $S_INSTALL_PEO  $S_CONFIGURE_MPD $S_END"

FULL_INSTALL_WITHOUT_LIBXML2_MPICH2="$S_INTRODUCTION $S_INSTALL_EO $S_INSTALL_MO $S_INSTALL_MOEO $S_INSTALL_PEO  $S_CONFIGURE_MPD $S_END_WITHOUT_INFO"

# basic install
BASIC_INSTALL="$S_INTRODUCTION $S_INSTALL_EO $S_INSTALL_MO $S_INSTALL_MOEO $S_END"

# install only paradiseo-peo
PARALLEL_INSTALL="$S_PEO_CHECK $S_INTRODUCTION $S_UNPACK_LIBXML $S_CHECK_AUTOTOOLS $S_INSTALL_LIBXML $S_REMOVE_TEMP_LIBXML $S_UNPACK_MPICH $S_INSTALL_MPICH $S_REMOVE_TEMP_MPICH $S_CONFIGURE_ENV $S_INSTALL_PEO $S_CONFIGURE_MPD $S_END"

PARALLEL_INSTALL_WITHOUT_LIBXML2="$S_PEO_CHECK $S_INTRODUCTION  $S_UNPACK_MPICH $S_CHECK_AUTOTOOLS $S_INSTALL_MPICH $S_REMOVE_TEMP_MPICH $S_CONFIGURE_MPICH_ENV $S_INSTALL_PEO $S_CONFIGURE_MPD $S_END"

PARALLEL_INSTALL_WITHOUT_MPICH2="$S_PEO_CHECK $S_INTRODUCTION $S_CHECK_AUTOTOOLS $S_UNPACK_LIBXML $S_INSTALL_LIBXML $S_REMOVE_TEMP_LIBXML $S_CONFIGURE_LIBXML2_ENV $S_INSTALL_PEO $S_CONFIGURE_MPD $S_END"

PARALLEL_INSTALL_WITHOUT_LIBXML2_MPICH2="$S_PEO_CHECK $S_INTRODUCTION $S_INSTALL_PEO $S_CONFIGURE_MPD $S_END_WITHOUT_INFO"

# remove a previous install
RM_PREVIOUS_INSTALL="$S_REMOVE_INSTALL"

#others
LIBS_PATH=lib
LIBXML2_ARCHIVE=libxml2-2.6.0
LIBXML2_ARCHIVE_SUFFIX=.tar.bz2
MPICH2_ARCHIVE=mpich2-1.0.3
MPICH2_ARCHIVE_SUFFIX=.tar.gz
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
BASIC_INSTALL_MISSING_ERROR=121
DART_SUBMISSION_ERROR=64
CHECK_AUTOTOOLS_ERROR=122

#Date
DATE=`/bin/date '+%Y%m%d%H%M%S'`
# create log file
SPY=$PWD/logs/install-paradiseo.${DATE}.log

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
		return ${RETURN_CODE}
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

	$CHECK_AUTOTOOLS_ERROR)
		echo
		echo "  An error has occured : cannot find the Autotools. See $SPY for more details" 
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

	$BASIC_INSTALL_MISSING_ERROR)
		echo
		echo "  An error has occured : impossible to find the basic install of ParadisEO. See $SPY for more details" 
		echo " You may not have a basic ParadisEO install available in the current directory"
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
		echo "Installing the environment for ParadisEO...Note that the librairies \"libxml2\" and \"mpich2\" required for ParadisEO are provided with this package. To avoid build and test reports to be sent to our repository, please stop the program and restart it using the --skipdart option."
		sleep 6	
		echo
		echo
		return $SUCCESSFUL_STEP
		;;

	$S_UNPACK_LIBXML)
		########## unpacking libxml2 ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Unpacking libxml2 (required for ParadisEO) ...'
		
		execute_cmd "rm -Rf $installKitPath/$LIBXML2_ARCHIVE_SUFFIX" "[$currentStepCounter-1] Remove potential existing dir $installKitPath/$LIBXML2_ARCHIVE"  $SPY 
		RETURN=$?

		execute_cmd "tar xvjf $resourceKitPath/$LIBS_PATH/$LIBXML2_ARCHIVE$LIBXML2_ARCHIVE_SUFFIX  --directory $installKitPath" "[$currentStepCounter-2] Unpack Libxml2" $SPY
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
		
		execute_cmd "rm -Rf $installKitPath/$MPICH2_ARCHIVE" "[$currentStepCounter-1] Remove potential existing dir $installKitPath/$MPICH2_ARCHIVE"  $SPY 
		RETURN=$?

		execute_cmd "tar xzvf $resourceKitPath/$LIBS_PATH/$MPICH2_ARCHIVE$MPICH2_ARCHIVE_SUFFIX --directory $installKitPath" "[$currentStepCounter-2] Unpack Mpich2" $SPY
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

		if [ ! "$installKitPath" = "$resourceKitPath" ]
		    then
		    cp  -Rf $resourceKitPath/paradiseo-eo/ $installKitPath/
		    rm -Rf $installKitPath/paradiseo-eo/build
		fi

		execute_cmd "mkdir $installKitPath/paradiseo-eo/build" "[$currentStepCounter-1] Create build directory"  $SPY 	
		
		execute_cmd "cd $installKitPath/paradiseo-eo/build" "[$currentStepCounter-2] Go in Paradiseo-EO build dir"  $SPY 
		RETURN=`expr $RETURN + $?`
		
		execute_cmd " echo \"cmake ../  -G$BUILD_PROCESS_TYPE $OPTIONNAL_CMAKE_FLAGS\" " "[$currentStepCounter-3] Run CMake using generator $BUILD_PROCESS_TYPE"  $SPY
		
		cmake ../  -G"$BUILD_PROCESS_TYPE"  -DCMAKE_BUILD_TYPE=$BUILD_TYPE $OPTIONNAL_CMAKE_FLAGS >> ${SPY} 2>> ${SPY}
		RETURN=`expr $RETURN + $?`

		if [ "$COMPILE_PARADISEO" = "1" ] 
		then
			execute_cmd "ctest $CTEST_CONFIG" "[$currentStepCounter-4] Compile ParadisEO-EO using CTest"  $SPY
			LAST_RETURN=$?
			# don't consider a submission error as a "right error"
			if [ ! "$LAST_RETURN" = "$DART_SUBMISSION_ERROR" ]
			then
				RETURN=`expr $RETURN + $LAST_RETURN`
			fi			
		fi
				
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
	
		if [ ! "$installKitPath" = "$resourceKitPath" ]
		    then
		    cp  -Rf $resourceKitPath/paradiseo-mo/ $installKitPath/
		    cp $resourceKitPath/install.cmake $installKitPath/
		    rm -Rf $installKitPath/paradiseo-mo/build/*
		fi

		execute_cmd "cd $installKitPath/paradiseo-mo/build" "[$currentStepCounter-1] Go in Paradiseo-MO dir"  $SPY 
		RETURN=$?
		
		execute_cmd " echo \"cmake ../ -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE -G\"$BUILD_PROCESS_TYPE\" $OPTIONNAL_CMAKE_FLAGS -DCMAKE_BUILD_TYPE=$BUILD_TYPE\" " "[$currentStepCounter-2] Run CMake using generator $BUILD_PROCESS_TYPE -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE"  $SPY
		cmake ../ -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE -G"$BUILD_PROCESS_TYPE" -DCMAKE_BUILD_TYPE=$BUILD_TYPE $OPTIONNAL_CMAKE_FLAGS>> ${SPY} 2>> ${SPY}
		RETURN=`expr $RETURN + $?`
		
		if [ "$COMPILE_PARADISEO" = "1" ] 
		then
			execute_cmd "ctest $CTEST_CONFIG" "[$currentStepCounter-3] Compile ParadisEO-MO using CTest"  $SPY	
			LAST_RETURN=$?
			# don't consider a submission error as a "right error"
			if [ ! "$LAST_RETURN" = "$DART_SUBMISSION_ERROR" ]
			then
				RETURN=`expr $RETURN + $LAST_RETURN`
			fi			
			
			execute_cmd "make install" "[$currentStepCounter-4] Make install of ParadisEO-MO"  $SPY
			RETURN=`expr $RETURN + $?`
		fi
		
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
		
		if [ ! "$installKitPath" = "$resourceKitPath" ]
		    then
		    cp  -Rf $resourceKitPath/paradiseo-moeo/ $installKitPath/
		    rm -Rf $installKitPath/paradiseo-moeo/build/*
		fi

		execute_cmd "cd $installKitPath/paradiseo-moeo/build" "[$currentStepCounter-1] Go in Paradiseo-MOEO dir"  $SPY 
		RETURN=$?
		
		execute_cmd " echo \"cmake ../ -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE -G\"$BUILD_PROCESS_TYPE\" -DCMAKE_BUILD_TYPE=$BUILD_TYPE $OPTIONNAL_CMAKE_FLAGS \"" "[$currentStepCounter-2] Run CMake using generator $BUILD_PROCESS_TYPE -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE"  $SPY
		cmake ../ -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE -G"$BUILD_PROCESS_TYPE" -DCMAKE_BUILD_TYPE=$BUILD_TYPE $OPTIONNAL_CMAKE_FLAGS >> ${SPY} 2>> ${SPY}
		RETURN=`expr $RETURN + $?`		
		
		if [ "$COMPILE_PARADISEO" = "1" ] 
		then
			execute_cmd "ctest $CTEST_CONFIG" "[$currentStepCounter-3] Compile ParadisEO-MOEO using CTest"  $SPY
			LAST_RETURN=$?
			# don't consider a submission error as a "right error"
			if [ ! "$LAST_RETURN" = "$DART_SUBMISSION_ERROR" ]
			then
				RETURN=`expr $RETURN + $LAST_RETURN`
			fi	
			execute_cmd "make install" "[$currentStepCounter-4] Make install ParadisEO-MOEO"  $SPY
			RETURN=`expr $RETURN + $?`
		fi
		
		# Make link  with the install.cmake (at the top level)
		#execute_cmd "ln -s $installKitPath/install.cmake $installKitPath/paradiseo-moeo/install.cmake" "[$currentStepCounter-5] Create link to install.cmake for MOEO"  $SPY 
		
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
		
	$S_CHECK_AUTOTOOLS)
		########## Check if we can user Autogen & Autoconf (only for libxml2 & mpich2 installation) ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Looking for the Autotools (Autogen & Autoconf, required to install libxml2 and mpich2)'
		
		(autoconf --version) < /dev/null > /dev/null 2>&1 ||
		{
			echo ''
			echo "		--> Error when checking for Autoconf"
		    echo "		You must have Autoconf installed to compile $PROG. Please update your system to get it before installing $PROG."
		    execute_cmd "autoconf --version" "[$currentStepCounter-1] Check Autoconf" $SPY    
		   	echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		    return $CHECK_AUTOTOOLS_ERROR
		}
		
		(automake --version) < /dev/null > /dev/null 2>&1 ||
		{
			echo ''
			echo "		--> Error when checking for Automake"
		    echo "		You must have Automake installed to compile $PROG. Please update your system to get it before installing $PROG."
		    execute_cmd "automake --version" "[$currentStepCounter-2] Check Automake" $SPY 
		    echo -e ' \033[40m\033[1;33m### END ### \033[0m '
		    return $CHECK_AUTOTOOLS_ERROR
		}
	
		echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
		echo
		return $SUCCESSFUL_STEP
		;;
		
	$S_INSTALL_LIBXML)
		########## installing LIBXML2 ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Installing libxml2. Please wait ...'
		
		execute_cmd "rm -Rf $installKitPath/libxml2" "[$currentStepCounter-0] Remove potential existing dir $installKitPath/libxml2"  $SPY 
		RETURN=$?

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
		
		execute_cmd "rm -Rf $installKitPath/mpich2" "[$currentStepCounter-0] Remove potential existing dir $installKitPath/mpich2"  $SPY 
		RETURN=$?

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

		execute_cmd "rm -Rf $installKitPath/paradiseo-eo/build/*" "[$currentStepCounter] Remove $installKitPath/paradiseo-eo/build/*" $SPY 
		idx=`expr $idx + $?`
		execute_cmd "rm -Rf $installKitPath/paradiseo-mo/build/*" "[$currentStepCounter] Remove $installKitPath/paradiseo-mo/build/*" $SPY 
		idx=`expr $idx + $?`
		execute_cmd "rm -Rf $installKitPath/paradiseo-moeo/build/*" "[$currentStepCounter] Remove $installKitPath/paradiseo-moeo/build/*" $SPY 
		idx=`expr $idx + $?`
		execute_cmd "rm -Rf $installKitPath/paradiseo-peo/build/*" "[$currentStepCounter] Remove $installKitPath/paradiseo-peo/build/*" $SPY 
		idx=`expr $idx + $?`
		
		if [ ! $(($idx)) = 0 ]
		then
			echo ''
			echo "		--> Error when removing previous install of ParadisEO"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $RM_UTIL_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;

	$S_CONFIGURE_MPICH_ENV)
		########## Configuring mpich environment variables ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Configuring environment variables for mpich2 ...'

		execute_cmd "export PATH=$PATH:`xml2-config --prefix`/bin:$installKitPath/mpich2/bin" "[$currentStepCounter-2] Export PATH variable" $SPY 
		idx=$?	

		execute_cmd "echo export PATH=$PATH:$installKitPath/mpich2/bin" "[$currentStepCounter-4] Export PATH variable into env" $SPY $HOME/.bashrc
		idx=`expr $idx + $?`

		execute_cmd "source $HOME/.bashrc" "[$currentStepCounter-5] Export variables for mpich2" $SPY
		idx=`expr $idx + $?`

		if [ ! $(($idx)) = 0 ]
		then
			echo ''
			echo "		--> Error when configuring environment variables for mpich2"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $VAR_CONFIG_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;

	$S_CONFIGURE_LIBXML2_ENV)
		########## Configuring environment variables ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Configuring environment variables for libxml2 ...'
		
		execute_cmd "XML2_CONFIG=\`xml2-config --prefix\`" "[$currentStepCounter-1] Run xml2-config variable" $SPY
		idx=$?
		echo "******** $XML2_CONFIG *********"

		execute_cmd "export LD_LIBRARY_PATH=`xml2-config --prefix`/lib:" "[$currentStepCounter-2] Export LD_LIBRARY_PATH variable" $SPY
		idx=$?	 
		
		execute_cmd "echo export LD_LIBRARY_PATH=$`xml2-config --prefix`/lib" "[$currentStepCounter-3] Export LD_LIBRARY_PATH variable into env" $SPY $HOME/.bashrc
		idx=$?	 

		execute_cmd "echo export PATH=$PATH:`xml2-config --prefix`/bin" "[$currentStepCounter-4] Export PATH variable into env" $SPY $HOME/.bashrc
		idx=`expr $idx + $?`

		execute_cmd "source $HOME/.bashrc" "[$currentStepCounter-5] Export variables for libxml2" $SPY
		idx=`expr $idx + $?`

		if [ ! $(($idx)) = 0 ]
		then
			echo ''
			echo "		--> Error when configuring environment variables for libxml2"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $VAR_CONFIG_ERROR
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
		
		execute_cmd "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$installKitPath/libxml2/lib:" "[$currentStepCounter-1] Export LD_LIBRARY_PATH variable" $SPY
		idx=$?	 
		execute_cmd "export PATH=$PATH:$installKitPath/libxml2/bin:$installKitPath/mpich2/bin" "[$currentStepCounter-2] Export PATH variable" $SPY 
	
		execute_cmd "echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$installKitPath/libxml2/lib" "[$currentStepCounter-3] Export LD_LIBRARY_PATH variable into env" $SPY $HOME/.bashrc
		idx=$?	 

		execute_cmd "echo export PATH=$PATH:$installKitPath/libxml2/bin:$installKitPath/mpich2/bin" "[$currentStepCounter-4] Export PATH variable into env" $SPY $HOME/.bashrc
		idx=`expr $idx + $?`

		execute_cmd "source $HOME/.bashrc" "[$currentStepCounter-5] Export variables" $SPY
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
		
		if [ ! "$installKitPath" = "$resourceKitPath" ]
		    then
		    cp  -Rf $resourceKitPath/paradiseo-peo/ $installKitPath/
		    rm -Rf $installKitPath/paradiseo-peo/build/*
		fi

		execute_cmd "cd $installKitPath/paradiseo-peo/build" "[$currentStepCounter-1] Go in Paradiseo-PEO dir"  $SPY 
		RETURN=$?

		execute_cmd " echo \"cmake ../  -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE -G\"$BUILD_PROCESS_TYPE\" $OPTIONNAL_CMAKE_FLAGS \"" "[$currentStepCounter-2] Run CMake using generator $BUILD_PROCESS_TYPE -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE"  $SPY
		cmake ../  -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE -G"$BUILD_PROCESS_TYPE" $OPTIONNAL_CMAKE_FLAGS -DCMAKE_BUILD_TYPE=$BUILD_TYPE >> ${SPY} 2>> ${SPY}
		RETURN=`expr $RETURN + $?`
		
		if [ "$COMPILE_PARADISEO" = "1" ] 
		then
			execute_cmd "ctest $CTEST_CONFIG" "[$currentStepCounter-3] Compile ParadisEO-PEO using CTest"  $SPY
			LAST_RETURN=$?
			# don't consider a submission error as a "right error"
			if [ ! "$LAST_RETURN" = "$DART_SUBMISSION_ERROR" ]
			then
				RETURN=`expr $RETURN + $LAST_RETURN`
			fi	
			execute_cmd "make install" "[$currentStepCounter-4] Make install ParadisEO-PEO "  $SPY
			RETURN=`expr $RETURN + $?`
		fi
		
		# Make link  with the install.cmake (at the top level)
		#execute_cmd "ln -s $installKitPath/install.cmake $installKitPath/paradiseo-peo/install.cmake" "[$currentStepCounter-5] Create link to install.cmake for PEO"  $SPY 
		
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
		if [ -d paradiseo-eo -a -d paradiseo-mo -a -d paradiseo-moeo ]
		then
			if  [ -d libxml2 -o -d mpich2 ]
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
			return $BASIC_INSTALL_MISSING_ERROR	
		fi
		;;
	$S_END_WITHOUT_INFO)
		echo
		echo -e "	\033[40m\033[1;34m#  SUCCESSFULL INSTALLATION. \033[0m"
		echo
		return $SUCCESSFUL_STEP
		;;
	$S_END)
		echo -e "The file \".bashrc\" file located in your directory $HOME has been MODIFIED. The following variables have been modified at the end:"
		echo -e " LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$installKitPath/libxml2/lib: "
		echo -e " PATH=\$PATH:$installKitPath/libxml2/bin:$installKitPath/mpich2/bin"
		echo -e "These variables are necessary to compile any program using ParadisEO-PEO.\033[40m\033[1;33m If you want to keep them in your environment in order not to have to set them each time you compile, enter  \"source $HOME/.bashrc\" \033[0m. If you don't want to use these variables, please remove them from $HOME/.bashrc."
		sleep 2
		echo
		echo
		echo -e "	\033[40m\033[1;34m#  SUCCESSFULL INSTALLATION. \033[0m"
		echo
		return $SUCCESSFUL_STEP
		;;
	*)
		
	;;
	esac
}


#------------------------------------------------------#
#-- FUNCTION   :  check_utils_install		    ---#
#------------------------------------------------------#
#-- PARAMETERS : No 				    ---#	
#-- Set some global variables (used for libxml2     ---#
#-- and mpich2 install management)                  ---#
#------------------------------------------------------#
function check_utils_install
{
	# is there an available version of mpich on the system ?
	(mpicxx --version) < /dev/null > /dev/null 2>&1 ||
	{
	IS_MPICH_INSTALLED=0
	}
	
	# is there an available version of libxml2 on the system ?
	(xml2-config --version) < /dev/null > /dev/null 2>&1 ||
	{
	IS_LIBXML2_INSTALLED=0
	}
	
	# ask the user if he'd like to keep his mpich version instead of the provided one
	if [ "$IS_MPICH_INSTALLED" = "1" ]
	then
		echo 
		echo -e ' \033[40m\033[1;31m###  A version of the MPI compiler has been detected on your system. Do you want to use it (if no, the mpich2 package, provided with ParadisEO, will be installed on your computer) [y/n] \033[0m '
	
		execute_cmd "echo \" A version of the MPI compiler has been detected on the system\"" "Is there a previous install of MPI ?" $SPY
	
		TREATENED=0
		while [ "$TREATENED" = "0" ]
		do
			read MPICH_YES_NO
			if  [ ! "$MPICH_YES_NO" = "y" ] && [ ! "$MPICH_YES_NO" = "n" ] 
			then
				TREATENED=0
			else
				if [ "$MPICH_YES_NO" = "y" ] 
				then
					USE_EXISTING_MPICH=1
				else
					USE_EXISTING_MPICH=0
				fi	
				TREATENED=1
			fi
		done
	fi
	
	
	# ask the user if he'd like to keep his libxml2 version instead of the provided one
	if [ "$IS_LIBXML2_INSTALLED" = "1" ]
	then
		echo 
		echo -e ' \033[40m\033[1;31m###  A version of libxml2 has been detected on your system. Do you want to use it (if no, the libxml2 package, provided with ParadisEO, will be installed on your computer) [y/n] \033[0m '
	
		execute_cmd "echo \" A version of libxml2 has been detected on the system\"" "Is there a previous install of libxml2 ?" $SPY
	
		TREATENED=0
		while [ "$TREATENED" = "0" ]
		do
			read LIBXML2_YES_NO
			if [ ! "$LIBXML2_YES_NO" = "y" ] && [ ! "$LIBXML2_YES_NO" = "n" ] 
			then
				TREATENED=0
			else
				if [ "$LIBXML2_YES_NO" = "y" ] 
				then
					USE_EXISTING_LIBXML2=1
				else
					USE_EXISTING_LIBXML2=0
				fi	
				TREATENED=1
			fi
		done
	fi

}



########################################################
######### 		BODY 							########
#########################################################


(cmake --version) < /dev/null > /dev/null 2>&1 ||
{
    echo "You must have CMake installed to compile $PROG. Please update your system to get it before installing $PROG."
    execute_cmd "echo \"You must have CMake installed to compile $PROG. Please update your system to get it before installing $PROG.\"" "[0-3] Check autoconf" $SPY 
    DIE=1
}

if [ "$DIE" = "1" ] 
then
    exit 1
fi



########################################################################
# Simple menu
# The options are :
#  --prefix
#  --debug
#  --skipdart
#  --help
#######################################################################

INSTALL_TREATENED=0
INSTALL_PATH=$PWD
for i in $*
do
  if [ "${i%=*}" = "--help" ] || [ "${i%=*}" = "-h" ]
  then
  		clear
  		echo "installParadiseo.sh"
		echo
		echo -e "\033[1mNAME\033[0m"
		echo '	installParadiseo.sh  - Install ParadisEO'
		echo
		echo -e "\033[1mSYNOPSIS\033[0m"
		echo -e '	\033[1m./installParadiseo.sh\033[0m  or \033[1mbash installParadiseo.sh\033[0m'
		echo -e '	[\033[1m--prefix=\033[0m\033[4mPATH\033[0m] [\033[1m--debug\033[0m] [\033[1m--skipdart\033[0m] [\033[1m--home=\033[0m\033[4mHOME\033[0m] [\033[1m-h\033[0m] [\033[1m--help\033[0m]'
		echo
		echo -e "\033[1mDESCRIPTION\033[0m"
		echo -e "	\033[1m--prefix=\033[0m\033[4mPATH\033[0m"
		echo -e "		ParadisEO will be installed in the directory \033[0m\033[4mPATH\033[0m. The current directory is used by default."
		echo
		echo -e "	\033[1m--debug\033[0m"
		echo '		Debug mode, set warning compiler flags and run tests.'
		echo
		echo -e "	\033[1m--skipdart\033[0m"
		echo '		Use this option to avoid build/test report submission to our Dart server.'
		echo		
		echo -e "	\033[1m--home=\033[0m\033[4mHOME\033[0m"
		echo -e "		Using \033[0m\033[4mHOME\033[0m as your home directory. Should be used when ~ doesnt reference your home. "
		echo			
		echo -e "	\033[1m-h, --help\033[0m"
		echo '		Print these useful lines.'
		echo	
		echo -e "\033[1mAUTHOR\033[0m"
		echo  "	Written by Thomas Legrand."
		echo
		echo -e "\033[1mBUGS\033[0m"
		echo  "	Report bugs to paradiseo-bugs@lists.gforge.inria.fr."
		echo
		echo -e "\033[1mCOPYRIGHT\033[0m"
		echo "	This software is governed by the CeCILL license under French law and"
		echo "	abiding by the rules of distribution of free software.  You can  use,"
		echo "	modify and/ or redistribute the software under the terms of the CeCILL"
		echo "	license as circulated by CEA, CNRS and INRIA at the following URL"
		echo "	http://www.cecill.info. "
		echo
		echo -e "\033[1mSEE ALSO\033[0m"		
		echo  "	For further help, please contact paradiseo-help@lists.gforge.inria.fr."	
		echo
		exit
  fi
  if [ "${i%=*}" = "--prefix" ]
   then
      INSTALL_PATH=${i#*=}
  fi
  if [ "${i%=*}" = "--debug" ]
   then
      BUILD_TYPE=Debug
      OPTIONNAL_CMAKE_FLAGS='-DENABLE_CMAKE_TESTING=TRUE'
      CTEST_CONFIG="$CTEST_CONFIG -D ExperimentalTest"
  fi
  if [ "${i%=*}" = "--skipdart" ]
   then
  	  SKIP_DART="1"
  fi
  if [ "${i%=*}" = "--home" ]
   then
  	 HOME_PATH=${i#*=}
  fi
done
#######################################################################

### Do we have a valid home path ?
if [ ! -d $HOME_PATH ]
then
	echo " Please give a valid path for your home directory (use --help for further information)"
fi


### Add a CTest flag depending on the "skipdart" option.
if [ ! "$SKIP_DART" = "1" ]
then
  	  CTEST_CONFIG="$CTEST_CONFIG -D ExperimentalSubmit"
fi
  
 
### Need the generator
BUILD_PROCESS_TYPE=0
GENERATOR_TREATENED=0

while [ ! "$GENERATOR_TREATENED" = "1" ]
do	
	case "$BUILD_PROCESS_TYPE" in
	
	$P_UNIX_MAKEFILES)
		BUILD_PROCESS_TYPE="$G_UNIX_MAKEFILES"
		GENERATOR_TREATENED=1		
		;;
	
	$P_KDEVELOP3_PROJECT)
		BUILD_PROCESS_TYPE="$G_KDEVELOP3_PROJECT"
		GENERATOR_TREATENED=1
		COMPILE_PARADISEO=1
		;;
	
	$P_XCODE_PROJECT)
		BUILD_PROCESS_TYPE="$G_XCODE_PROJECT"
		GENERATOR_TREATENED=1
		COMPILE_PARADISEO=1
		;;
	*)
		echo
		echo -e ' \033[40m\033[1;33m### Please select the kind of "Makefile" you want to generate: ### \033[0m '
		echo
		echo "	 $P_UNIX_MAKEFILES : Unix Makefiles (standard Makefiles)"
		echo "	 $P_KDEVELOP3_PROJECT : KDevelop3 project files"
		echo "	 $P_XCODE_PROJECT : Xcode project files"
		read BUILD_PROCESS_TYPE
	;;
	esac
done


while [ ! "$INSTALL_TREATENED" = "1" ]
do	
	case "$INSTALL_TYPE" in
	$P_FULL_INSTALL)
	
		check_utils_install

		if [ "$USE_EXISTING_MPICH" = "1" ] && [ "$USE_EXISTING_LIBXML2" = "1" ] 
		then
			THE_GOOD_INSTALL=$FULL_INSTALL_WITHOUT_LIBXML2_MPICH2
		
		elif [ "$USE_EXISTING_MPICH" = "1" ] && [ "$USE_EXISTING_LIBXML2" = "0" ] 
		then
			THE_GOOD_INSTALL=$FULL_INSTALL_WITHOUT_MPICH
		
		elif [ "$USE_EXISTING_MPICH" = "0" ] && [ "$USE_EXISTING_LIBXML2" = "1" ] 
		then
			THE_GOOD_INSTALL=$FULL_INSTALL_WITHOUT_LIBXML2
		
		elif [ "$USE_EXISTING_MPICH" = "0" ] && [ "$USE_EXISTING_LIBXML2" = "0" ] 
		then
			THE_GOOD_INSTALL=$FULL_INSTALL
		else
			THE_GOOD_INSTALL=$FULL_INSTALL
		fi

		counter=0
		for step in $THE_GOOD_INSTALL	
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

		check_utils_install

		if [ "$USE_EXISTING_MPICH" = "1" ] && [ "$USE_EXISTING_LIBXML2" = "1" ] 
		then
			THE_GOOD_PARALLEL_INSTALL=$PARALLEL_INSTALL_WITHOUT_LIBXML2_MPICH2
		
		elif [ "$USE_EXISTING_MPICH" = "1" ] && [ "$USE_EXISTING_LIBXML2" = "0" ] 
		then
			THE_GOOD_PARALLEL_INSTALL=$PARALLEL_INSTALL_WITHOUT_MPICH2
		
		elif [ "$USE_EXISTING_MPICH" = "0" ] && [ "$USE_EXISTING_LIBXML2" = "1" ] 
		then
			THE_GOOD_PARALLEL_INSTALL=$PARALLEL_INSTALL_WITHOUT_LIBXML2
		
		elif [ "$USE_EXISTING_MPICH" = "0" ] && [ "$USE_EXISTING_LIBXML2" = "0" ] 
		then
			THE_GOOD_PARALLEL_INSTALL=$PARALLEL_INSTALL
		else
			THE_GOOD_PARALLEL_INSTALL=$PARALLEL_INSTALL
		fi

		counter=0
		for step in $THE_GOOD_PARALLEL_INSTALL	
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




