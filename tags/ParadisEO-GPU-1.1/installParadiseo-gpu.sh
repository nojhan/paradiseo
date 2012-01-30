#!/bin/sh

#########################################################################################
#
#	Project		:	paradisEO-GPU full package install
#	File		:	installParadiseo-gpu.sh
#	Comment	:	# This file attempts to install ParadisEO-GPU #
#
#########################################################################################

# global variables
installStartTime=$SECONDS 
resourceKitPath=$PWD
TAR_MSG=" "
DIE=0
PROG=ParadisEO-GPU
CMAKE_PRIMARY_CONFIG_FILE=install.cmake
HOME_PATH=$HOME
CUDA=" " #****
bash_path='$PATH'
library_path='$LD_LIBRARY_PATH'

# generator types available on Unix platforms
P_UNIX_MAKEFILES=1
G_UNIX_MAKEFILES="Unix Makefiles"

# should we compile ParadisEO ?
COMPILE_PARADISEO=1

# Build types
DEFAULT_BUILD_TYPE=Release
BUILD_TYPE=$DEFAULT_BUILD_TYPE

# CMake/CTest/Dart flags
CTEST_DEFAULT_CONFIG="-D ExperimentalStart -D ExperimentalBuild -D ExperimentalTest"
CTEST_CONFIG=$CTEST_DEFAULT_CONFIG

# What are the tests that should be always run ?
MIN_CMAKE_FLAGS='-DENABLE_MINIMAL_CMAKE_TESTING=TRUE'

# install types to select in the main menu
P_FULL_INSTALL=1
P_RM_PREVIOUS_INSTALLL=2
P_EXIT_INSTALL=3

IS_CUDA_INSTALLED=1 

# install steps
S_INTRODUCTION=1000
S_UNPACK_EO=1001
S_INSTALL_EO=1002
S_INSTALL_MO=1003
S_INSTALL_CUDA=1004 #****
S_INSTALL_GPU=1005 #****
S_CONFIGURE_ENV=1006
S_REMOVE_INSTALL=1007
S_END=1008
#S_CHECK_AUTOTOOLS=1018

#### define what are the possible installs and their content

# full install

FULL_INSTALL="$S_INTRODUCTION $S_INSTALL_EO $S_INSTALL_MO $S_INSTALL_CUDA $S_INSTALL_GPU $S_END"

# remove a previous install
RM_PREVIOUS_INSTALL="$S_REMOVE_INSTALL"

#others
LIBS_PATH=lib
# errors
SUCCESSFUL_STEP=0
EO_UNPACKING_ERROR=100
EO_INSTALL_ERROR=104
MO_INSTALL_ERROR=108
GPU_INSTALL_ERROR=112 #****
CUDA_INSTALLING_ERROR=116 #****
REMOVE_TEMP_DIRECTORY_ERROR=113
VAR_CONFIG_ERROR=114
RM_PARADISEO_EO_ERROR=119
DART_SUBMISSION_ERROR=64

#Date
DATE=`/bin/date '+%Y%m%d%H%M%S'`
# create log file
SPY=$PWD/logs/install-paradiseo-gpu.${DATE}.log

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
        $CUDA_INSTALLING_ERROR)
		echo
		echo "  An error has occured : impossible to install CudaToolkit.See $SPY for more details" 
	  	echo "If you need help, please contact paradiseo-help@lists.gforge.inria.fr and join $SPY"
		echo 
		echo 
		kill $$;;

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

	$RM_PARADISEO_EO_ERROR)
		echo
		echo "  An error has occured : impossible to remove ParadisEO-EO. See $SPY for more details" 
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
		echo -e ' \033[40m\033[1;33m### ParadisEO-GPU install starting .... ### \033[0m '
		echo
		echo "Installing the environment for ParadisEO-GPU... To avoid build and test reports to be sent to our repository, please stop the program and restart it using the --skipdart option."
		sleep 6	
		echo
		echo
		return $SUCCESSFUL_STEP
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
		
		execute_cmd " echo \"cmake ../  -G$BUILD_PROCESS_TYPE $MIN_CMAKE_FLAGS $OPTIONNAL_CMAKE_FLAGS\" " "[$currentStepCounter-3] Run CMake using generator $BUILD_PROCESS_TYPE"  $SPY
		
		cmake ../  -G"$BUILD_PROCESS_TYPE"  -DCMAKE_BUILD_TYPE=$BUILD_TYPE $MIN_CMAKE_FLAGS $OPTIONNAL_CMAKE_FLAGS >> ${SPY} 2>> ${SPY}
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
		
		execute_cmd " echo \"cmake ../ -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE -G\"$BUILD_PROCESS_TYPE\" $MIN_CMAKE_FLAGS $OPTIONNAL_CMAKE_FLAGS -DCMAKE_BUILD_TYPE=$BUILD_TYPE\" " "[$currentStepCounter-2] Run CMake using generator $BUILD_PROCESS_TYPE -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE"  $SPY
		cmake ../ -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE -G"$BUILD_PROCESS_TYPE" -DCMAKE_BUILD_TYPE=$BUILD_TYPE $MIN_CMAKE_FLAGS $OPTIONNAL_CMAKE_FLAGS>> ${SPY} 2>> ${SPY}
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

 	  $S_INSTALL_CUDA)
		########## installing cudaToolkit 3.2##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> installing cudaToolkit (required for ParadisEO-GPU) ...'
		execute_cmd "cd $installKitPath/downloads/" "[$currentStepCounter-2] Go in downloads dir"  $SPY 
                sudo ./cudatoolkit_3.2.16_linux_32_ubuntu10.04.run
		RETURN=$?
          if [ ! "$?" = "0" ]
		then
			echo ''
			echo "		--> Error when installing cudaToolkit"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $CUDA_INSTALLING_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		kill $?;;

  	$S_INSTALL_GPU)
		##########  installing paradiseo-gpu ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Installing Paradiseo-GPU. Please wait ...'
	
		if [ ! "$installKitPath" = "$resourceKitPath" ]
		    then
		    cp  -Rf $resourceKitPath/paradiseo-gpu/ $installKitPath/
		    cp $resourceKitPath/install.cmake $installKitPath/
		    rm -Rf $installKitPath/paradiseo-gpu/build/*
		fi

		execute_cmd "cd $installKitPath/paradiseo-gpu/build" "[$currentStepCounter-1] Go in Paradiseo-GPU dir"  $SPY 
		RETURN=$?
		
		execute_cmd " echo \"cmake ../ -G\"$BUILD_PROCESS_TYPE\" $MIN_CMAKE_FLAGS $OPTIONNAL_CMAKE_FLAGS -DCMAKE_BUILD_TYPE=$BUILD_TYPE\" " "[$currentStepCounter-2] Run CMake using generator $BUILD_PROCESS_TYPE"  $SPY
		cmake ../ -Dconfig=$installKitPath/$CMAKE_PRIMARY_CONFIG_FILE -G"$BUILD_PROCESS_TYPE" -DCMAKE_BUILD_TYPE=$BUILD_TYPE $MIN_CMAKE_FLAGS $OPTIONNAL_CMAKE_FLAGS>> ${SPY} 2>> ${SPY}
		RETURN=`expr $RETURN + $?`
		
		if [ "$COMPILE_PARADISEO" = "1" ] 
		then
			execute_cmd "ctest $CTEST_CONFIG" "[$currentStepCounter-3] Compile ParadisEO-GPU using CTest"  $SPY	
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
			echo "		--> Error when installing Paradiseo-GPU"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $GPU_INSTALL_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;

		
	$S_REMOVE_INSTALL)
		########## removing a previous install of EO ##########
		echo -e  "	\033[40m\033[1;34m# STEP $currentStepCounter \033[0m "
		echo '		--> Removing your previous install of ParadisEO-GPU ...'
		
		execute_cmd "rm -Rf $installKitPath/paradiseo-eo/build/*" "[$currentStepCounter] Remove $installKitPath/paradiseo-eo/build/*" $SPY 
		idx=$?
		execute_cmd "rm -Rf $installKitPath/paradiseo-mo/build/*" "[$currentStepCounter] Remove $installKitPath/paradiseo-mo/build/*" $SPY 
		idx=`expr $idx + $?`
		execute_cmd "rm -Rf $installKitPath/paradiseo-gpu/build/*" "[$currentStepCounter] Remove $installKitPath/paradiseo-gpu/build/*" $SPY 
		idx=`expr $idx + $?`
		
		if [ ! $(($idx)) = 0 ]
		then
			echo ''
			echo "		--> Error when removing previous install of ParadisEO"
			echo -e ' \033[40m\033[1;33m### END ### \033[0m '
			return $RM_UTIL_ERROR
		else
			echo -e "	\033[40m\033[1;34m# STEP $currentStepCounter OK \033[0m"
			echo -e "Please \033[40m\033[1;33m  CLOSE YOUR TERMINAL OR OPEN A NEW ONE \033[0m before proceeding with a new installation."
			echo
			return $SUCCESSFUL_STEP
		fi 
		;;

	$S_END)
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




########################################################
######### 		BODY 	                ########
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
  		echo "installParadiseo-gpu.sh"
		echo
		echo -e "\033[1mNAME\033[0m"
		echo '	installParadiseo-gpu.sh  - Install ParadisEO-GPU'
		echo
		echo -e "\033[1mSYNOPSIS\033[0m"
		echo -e '	\033[1m./installParadiseo-gpu.sh\033[0m  or \033[1mbash installParadiseo-gpu.sh\033[0m'
		echo -e '	[\033[1m--prefix=\033[0m\033[4mPATH\033[0m] [\033[1m--debug\033[0m] [\033[1m--skipdart\033[0m] [\033[1m--home=\033[0m\033[4mHOME\033[0m] [\033[1m-h\033[0m] [\033[1m--help\033[0m]'
		echo
		echo -e "\033[1mDESCRIPTION\033[0m"
		echo -e "	\033[1m--prefix=\033[0m\033[4mPATH\033[0m"
		echo -e "		ParadisEO-GPU will be installed in the directory \033[0m\033[4mPATH\033[0m. The current directory is used by default."
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
		echo  "	Written by Karima Boufaras."
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
BUILD_PROCESS_TYPE="$G_UNIX_MAKEFILES"
GENERATOR_TREATENED=1		
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
		echo "	 1 : Full install (all the components : EO,MO and GPU)"
		echo "	 2 : Remove a previous install of ParadisEO located in $INSTALL_PATH"
		echo "	 3 : Exit install"
		read INSTALL_TYPE
	;;
	esac
done
