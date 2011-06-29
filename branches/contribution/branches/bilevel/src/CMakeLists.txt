######################################################################################
### 1) Include the sources
######################################################################################
INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${MOEO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
######################################################################################


######################################################################################
### 2) Define your target(s): just the tsp lib here
######################################################################################
SET(VRP_LIB_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${VRP_LIB_OUTPUT_PATH})
SET (VRP_SOURCES 
	VRP2.h VRP2InitNN.h VRP2Eval.h 
	biVRPDistMat.h biVRPDistMat.cpp 
	mbiVRPDistMat.h mbiVRPDistMat.cpp 
	PLA.h biVRP.h biVRPUpEval.h mbiVRPLowEval.h biVRPLowEval.h 
	PLARandInit.h PLAInitResp.h biVRP.h biVRPInit.h biVRPPerf2Worth.h
	op/VRP2RBX.h op/VRP2SBX.h op/VRP2Repair.h op/VRPXover.h
	op/biVRPRBX.h op/biVRPSBX.h
	) 
SET (BEO_SOURCES  
	BEO.h beoEval.h beoCoevoOp.h beoLowQuad.h 
	beoUpQuad.h beoLowMut.h beoUpMut.h beoInit.h beoSelectOneUp.h 
	beoSelectOneLow.h beoCoevoPop.h beoDummyCoevoPop.h 
	beoCoevoAllPop.h beoLexContinue.h 
	beoRepairEval.h beoSelectCoevoPop.h beoSelectLow.h beoSelectUp.h 
	beoSelectOneLow.h beoSelectOneUp.h 
	beoBLAISArchivetemp.h beoBLAISArchivetemp.cpp
	op/beoRateContainerCoevoOp.h 
	op/beoBestExchange.h op/beoCoevoUpExchange.h 
	op/beoCoevoLowExchange.h op/beoLeftieExchange.h 
	op/beoRightieExchange.h beoLowComparator beoUpComparator
	)
ADD_LIBRARY(lvrp STATIC ${VRP_SOURCES})
ADD_LIBRARY(lbeo STATIC ${BEO_SOURCES})
######################################################################################
