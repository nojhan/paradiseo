IF(ENABLE_CMAKE_EXAMPLE OR CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_simpleHC.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_simpleHC.param
	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_firstImprHC.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_firstImprHC.param
	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_randomBestHC.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_randomBestHC.param
	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_neutralHC.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_neutralHC.param
	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_iterContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_iterContinuator.param
	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_fitContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_fitContinuator.param
	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_fullEvalContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_fullEvalContinuator.param
	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_evalContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_evalContinuator.param
	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_combinedContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_combinedContinuator.param
	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/firstImprHC_maxSAT.param
			${MO_BIN_DIR}/tutorial/Lesson1/firstImprHC_maxSAT.param
	)

	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson2/testNeighborhood.param
			${MO_BIN_DIR}/tutorial/Lesson2/testNeighborhood.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson3/testSimulatedAnnealing.param
			${MO_BIN_DIR}/tutorial/Lesson3/testSimulatedAnnealing.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson4/testSimpleTS.param
			${MO_BIN_DIR}/tutorial/Lesson4/testSimpleTS.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson5/testILS.param
			${MO_BIN_DIR}/tutorial/Lesson5/testILS.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/testRandomWalk.param
			${MO_BIN_DIR}/tutorial/Lesson6/testRandomWalk.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/testMetropolisHasting.param
			${MO_BIN_DIR}/tutorial/Lesson6/testMetropolisHasting.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/testRandomNeutralWalk.param
			${MO_BIN_DIR}/tutorial/Lesson6/testRandomNeutralWalk.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/sampling.param
			${MO_BIN_DIR}/tutorial/Lesson6/sampling.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/densityOfStates.param
			${MO_BIN_DIR}/tutorial/Lesson6/densityOfStates.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/autocorrelation.param
			${MO_BIN_DIR}/tutorial/Lesson6/autocorrelation.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/adaptiveWalks.param
			${MO_BIN_DIR}/tutorial/Lesson6/adaptiveWalks.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/fdc.param
			${MO_BIN_DIR}/tutorial/Lesson6/fdc.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/neutralDegree.param
			${MO_BIN_DIR}/tutorial/Lesson6/neutralDegree.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/fitnessCloud.param
			${MO_BIN_DIR}/tutorial/Lesson6/fitnessCloud.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/neutralWalk.param
			${MO_BIN_DIR}/tutorial/Lesson6/neutralWalk.param

	)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson7/hybridAlgo.param
			${MO_BIN_DIR}/tutorial/Lesson7/hybridAlgo.param

	)
		EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson9/VNS.param
			${MO_BIN_DIR}/tutorial/Lesson9/VNS.param

	)
	
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson10/testKswapNeighborhood.param
			${MO_BIN_DIR}/tutorial/Lesson10/testKswapNeighborhood.param

	)
	
ENDIF(ENABLE_CMAKE_EXAMPLE OR CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")

MACRO(ADD_COMMANDS_NEWMO)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson1/lesson1_simpleHC.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_simpleHC.param
			${MO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson1/lesson1_firstImprHC.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_firstImprHC.param
			${MO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson1/lesson1_randomBestHC.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_randomBestHC.param
			${MO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson1/lesson1_neutralHC.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_neutralHC.param
			${MO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson1/lesson1_iterContinuator.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_iterContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson1/lesson1_fitContinuator.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_fitContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson1/lesson1_fullEvalContinuator.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_fullEvalContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson1/lesson1_evalContinuator.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_evalContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson1/lesson1_combinedContinuator.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/lesson1_combinedContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson1/firstImprHC_maxSAT.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/firstImprHC_maxSAT.param
			${MO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson2/testNeighborhood.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson2/testNeighborhood.param
			${MO_BIN_DIR}/tutorial/Lesson2
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson3/testSimulatedAnnealing.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson3/testSimulatedAnnealing.param
			${MO_BIN_DIR}/tutorial/Lesson3
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson4/testSimpleTS.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson4/testSimpleTS.param
			${MO_BIN_DIR}/tutorial/Lesson4
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson5/testILS.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson5/testILS.param
			${MO_BIN_DIR}/tutorial/Lesson5
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/testRandomWalk.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/testRandomWalk.param
			${MO_BIN_DIR}/tutorial/Lesson6
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/testMetropolisHasting.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/testMetropolisHasting.param
			${MO_BIN_DIR}/tutorial/Lesson6
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/testRandomNeutralWalk.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/testRandomNeutralWalk.param
			${MO_BIN_DIR}/tutorial/Lesson6
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/sampling.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/sampling.param
			${MO_BIN_DIR}/tutorial/Lesson6
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/densityOfStates.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/densityOfStates.param
			${MO_BIN_DIR}/tutorial/Lesson6
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/fitnessCloud.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/fitnessCloud.param
			${MO_BIN_DIR}/tutorial/Lesson6
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/neutralWalk.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/neutralWalk.param
			${MO_BIN_DIR}/tutorial/Lesson6
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/autocorrelation.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/autocorrelation.param
			${MO_BIN_DIR}/tutorial/Lesson6
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/adaptiveWalks.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/adaptiveWalks.param
			${MO_BIN_DIR}/tutorial/Lesson6
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/fdc.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/fdc.param
			${MO_BIN_DIR}/tutorial/Lesson6
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/neutralDegree.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/neutralDegree.param
			${MO_BIN_DIR}/tutorial/Lesson6
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson7/hybridAlgo.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson7/hybridAlgo.param
			${MO_BIN_DIR}/tutorial/Lesson7
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson9/VNS.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson9/VNS.param
			${MO_BIN_DIR}/tutorial/Lesson9
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson10/testKswapNeighborhood.param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson10/testKswapNeighborhood.param
			${MO_BIN_DIR}/tutorial/Lesson10
	)
	
ENDMACRO(ADD_COMMANDS_NEWMO)

MACRO(ADD_TARGET_NEWMO n)
	IF(${n} STREQUAL "lesson1")
		ADD_CUSTOM_TARGET(lesson1 DEPENDS
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_simpleHC
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_simpleHC.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_firstImprHC
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_firstImprHC.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_randomBestHC
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_randomBestHC.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_neutralHC
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_neutralHC.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_iterContinuator
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_iterContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_fitContinuator
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_fitContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_fullEvalContinuator
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_fullEvalContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_evalContinuator
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_evalContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_combinedContinuator
			${MO_BIN_DIR}/tutorial/Lesson1/lesson1_combinedContinuator.param
			${MO_BIN_DIR}/tutorial/Lesson1/firstImprHC_maxSAT
			${MO_BIN_DIR}/tutorial/Lesson1/firstImprHC_maxSAT.param
		 )
	ELSEIF(${n} STREQUAL "lesson2")
		ADD_CUSTOM_TARGET(lesson2 DEPENDS
			${MO_BIN_DIR}/tutorial/Lesson2/testNeighborhood
			${MO_BIN_DIR}/tutorial/Lesson2/testNeighborhood.param
		)
	ELSEIF(${n} STREQUAL "lesson3")
		ADD_CUSTOM_TARGET(lesson3 DEPENDS
			${MO_BIN_DIR}/tutorial/Lesson3/testSimulatedAnnealing
			${MO_BIN_DIR}/tutorial/Lesson3/testSimulatedAnnealing.param
		)
	ELSEIF(${n} STREQUAL "lesson4")
		ADD_CUSTOM_TARGET(lesson4 DEPENDS
			${MO_BIN_DIR}/tutorial/Lesson4/testSimpleTS
			${MO_BIN_DIR}/tutorial/Lesson4/testSimpleTS.param
		)
	ELSEIF(${n} STREQUAL "lesson5")
		ADD_CUSTOM_TARGET(lesson5 DEPENDS
			${MO_BIN_DIR}/tutorial/Lesson5/testILS
			${MO_BIN_DIR}/tutorial/Lesson5/testILS.param
		)
	ELSEIF(${n} STREQUAL "lesson6")
		ADD_CUSTOM_TARGET(lesson6 DEPENDS
			${MO_BIN_DIR}/tutorial/Lesson6/testRandomWalk
			${MO_BIN_DIR}/tutorial/Lesson6/testRandomWalk.param
			${MO_BIN_DIR}/tutorial/Lesson6/testMetropolisHasting
			${MO_BIN_DIR}/tutorial/Lesson6/testMetropolisHasting.param
			${MO_BIN_DIR}/tutorial/Lesson6/testRandomNeutralWalk
			${MO_BIN_DIR}/tutorial/Lesson6/testRandomNeutralWalk.param
			${MO_BIN_DIR}/tutorial/Lesson6/sampling
			${MO_BIN_DIR}/tutorial/Lesson6/sampling.param
			${MO_BIN_DIR}/tutorial/Lesson6/densityOfStates
			${MO_BIN_DIR}/tutorial/Lesson6/densityOfStates.param
			${MO_BIN_DIR}/tutorial/Lesson6/autocorrelation
			${MO_BIN_DIR}/tutorial/Lesson6/autocorrelation.param
			${MO_BIN_DIR}/tutorial/Lesson6/adaptiveWalks
			${MO_BIN_DIR}/tutorial/Lesson6/adaptiveWalks.param
			${MO_BIN_DIR}/tutorial/Lesson6/fdc
			${MO_BIN_DIR}/tutorial/Lesson6/fdc.param
			${MO_BIN_DIR}/tutorial/Lesson6/neutralDegree
			${MO_BIN_DIR}/tutorial/Lesson6/neutralDegree.param
			${MO_BIN_DIR}/tutorial/Lesson6/fitnessCloud
			${MO_BIN_DIR}/tutorial/Lesson6/fitnessCloud.param
			${MO_BIN_DIR}/tutorial/Lesson6/neutralWalk
			${MO_BIN_DIR}/tutorial/Lesson6/neutralWalk.param
			
		)
	ELSEIF(${n} STREQUAL "lesson7")
		ADD_CUSTOM_TARGET(lesson7 DEPENDS
			${MO_BIN_DIR}/tutorial/Lesson7/hybridAlgo
			${MO_BIN_DIR}/tutorial/Lesson7/hybridAlgo.param
		)
	ELSEIF(${n} STREQUAL "lesson9")
		ADD_CUSTOM_TARGET(lesson9 DEPENDS
			${MO_BIN_DIR}/tutorial/Lesson9/VNS
			${MO_BIN_DIR}/tutorial/Lesson9/VNS.param
		)
    ELSEIF(${n} STREQUAL "lesson10")
		ADD_CUSTOM_TARGET(lesson10 DEPENDS
			${MO_BIN_DIR}/tutorial/Lesson10/testKswapNeighborhood
			${MO_BIN_DIR}/tutorial/Lesson10/testKswapNeighborhood.param
		)		
	ENDIF(${n} STREQUAL "lesson1")

ENDMACRO(ADD_TARGET_NEWMO)

ADD_SUBDIRECTORY(Lesson1)
ADD_SUBDIRECTORY(Lesson2)
ADD_SUBDIRECTORY(Lesson3)
ADD_SUBDIRECTORY(Lesson4)
ADD_SUBDIRECTORY(Lesson5)
ADD_SUBDIRECTORY(Lesson6)
ADD_SUBDIRECTORY(Lesson7)
ADD_SUBDIRECTORY(Lesson8)
ADD_SUBDIRECTORY(Lesson9)
ADD_SUBDIRECTORY(Lesson10)

IF(NOT CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" AND  NOT CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")
	ADD_COMMANDS_NEWMO()
	ADD_TARGET_NEWMO(install)
ENDIF(NOT CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" AND NOT CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")
