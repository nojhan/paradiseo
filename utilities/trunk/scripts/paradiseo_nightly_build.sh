#!/bin/sh

TEST_DIR=/opt/paradiseo/project-management/test/build/nightly

CMAKE_INSTALL_CONFIG=/opt/paradiseo/project-management/test/work-copy/nightly/trunk/install.cmake
PARADISEO_EO_DIR=/opt/paradiseo/project-management/test/work-copy/nightly/trunk/paradiseo-eo
PARADISEO_MO_DIR=/opt/paradiseo/project-management/test/work-copy/nightly/trunk/paradiseo-mo
PARADISEO_MOEO_DIR=/opt/paradiseo/project-management/test/work-copy/nightly/trunk/paradiseo-moeo
PARADISEO_PEO_DIR=/opt/paradiseo/project-management/test/work-copy/nightly/trunk/paradiseo-peo

EO_BUILD_TYPE=Debug
MO_BUILD_TYPE=Debug
MOEO_BUILD_TYPE=Debug
PEO_BUILD_TYPE=Debug

GENERATOR_LIST="Unix_Makefiles KDevelop3"

# export the ssh-agent variables
export SSH_AUTH_SOCK=/tmp/ssh-NFkaL18206/agent.18206
export SSH_AGENT_PID=18207

for gen in $GENERATOR_LIST	
do
	DATE=`/bin/date '+%Y%m%d%H%M%S'`
	SPY=$TEST_DIR/logs/nightly.${DATE}.log
		
	gen=`echo "$gen" | sed s/_/\ /g`
	echo "*** BEGIN Generator=$gen" >> $SPY
	

	### Remove build dirs content #########################################
	rm -Rf $PARADISEO_EO_DIR/build/CMakeCache.txt
	rm -Rf $PARADISEO_MO_DIR/build/CMakeCache.txt
	rm -Rf $PARADISEO_MOEO_DIR/build/CMakeCache.txt
	rm -Rf $PARADISEO_PEO_DIR/build/CMakeCache.txt

	################  EO ##################################################
	# Launch CMake for EO
	cd $PARADISEO_EO_DIR/build
	
	# Launch CTest for EO
	cmake .. -G"$gen" -DCMAKE_BUILD_TYPE=$EO_BUILD_TYPE -DENABLE_CMAKE_TESTING=TRUE >> $SPY
	ctest -D NightlyUpdate -D NightlyStart -D NightlyBuild -D NightlyCoverage  -D NightlyTest -D NightlyMemCheck -D NightlySubmit >> $SPY
	
	
	################  MO ##################################################
	# Launch CMake for MO
	cd $PARADISEO_MO_DIR/build
	
	# Launch CTest for MO
	cmake .. -Dconfig=$CMAKE_INSTALL_CONFIG -G"$gen" -DCMAKE_BUILD_TYPE=$MO_BUILD_TYPE -DENABLE_CMAKE_TESTING=TRUE >> $SPY
	ctest -D NightlyUpdate -D NightlyStart -D NightlyBuild -D NightlyCoverage  -D NightlyTest -D ightlyMemCheck -D NightlySubmit >> $SPY
	
	
	
	################  MOEO ##################################################
	# Launch CMake for MOEO
	cd $PARADISEO_MOEO_DIR/build
	
	# Launch CTest for MOEO
	cmake .. -Dconfig=$CMAKE_INSTALL_CONFIG -G"$gen" -DCMAKE_BUILD_TYPE=$MOEO_BUILD_TYPE -DENABLE_CMAKE_TESTING=TRUE >> $SPY
	ctest -D NightlyUpdate -D NightlyStart -D NightlyBuild -D NightlyCoverage  -D NightlyTest -D NightlyMemCheck -D NightlySubmit >> $SPY
	
	
	
	################  PEO ##################################################
	# Launch CMake for PEO
	cd $PARADISEO_PEO_DIR/build
	
	if [ ! -f ~/.mpd.conf ] 
	then
		echo "No mpd.conf file in ~ . Goind to create one ..." >> $SPY
		touch ~/.mpd.conf >> $SPY
		echo "MPD_SECRETWORD=hello-paradiseo" > ~/.mpd.conf
		echo "secretword=kiss-paradiseo" >> ~/.mpd.conf 
		chmod 600  ~/.mpd.conf >> $SPY
	fi

	# Launch CTest for PEO
	cmake .. -Dconfig=$CMAKE_INSTALL_CONFIG -G"$gen" -DCMAKE_BUILD_TYPE=$PEO_BUILD_TYPE -DENABLE_CMAKE_TESTING=TRUE >> $SPY
	ctest -D NightlyUpdate -D NightlyStart -D NightlyBuild -D NightlyCoverage  -D NightlyTest -D NightlyMemCheck -D NightlySubmit >> $SPY

	echo "*** END Generator=$gen" >> $SPY
done


