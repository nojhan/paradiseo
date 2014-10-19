#####################################################################################
### 0) Configuration files
#####################################################################################

# now create config headers
configure_file(config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)

# now create config install_symlink script file
configure_file(install_symlink.py.cmake ${CMAKE_CURRENT_BINARY_DIR}/install_symlink.py)

#####################################################################################

######################################################################################
### 1) Include the sources
######################################################################################

#include_directories(${EOMPI_SRC_DIR})
#include_directories(${EOSERIAL_SRC_DIR})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the eo target
######################################################################################

set(EO_LIB_OUTPUT_PATH ${EO_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${EO_LIB_OUTPUT_PATH})

set(EO_SOURCES
  eoFunctorStore.cpp
  eoPersistent.cpp
  eoPrintable.cpp
  eoCtrlCContinue.cpp
  eoScalarFitnessAssembled.cpp
  eoSIGContinue.cpp
  )

add_library(eo STATIC ${EO_SOURCES})

######################################################################################
### 3) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

set(EO_VERSION ${GLOBAL_VERSION})
set_target_properties(eo PROPERTIES VERSION "${EO_VERSION}")

install(TARGETS eo ARCHIVE DESTINATION ${LIB} COMPONENT libraries)

#file(GLOB HDRS *[^eo].h eo)
# Indication from CMake : "We do not recommend using GLOB to collect a list of 
#                         sources files from your source tree. If no 
#                         CMakeLists.txt file changes when a source is added or 
#                         removed then the generated build system cannot know 
#                         when to ask CMake to generate."

set(HDRS 
  apply.h
  eoAlgo.h
  eoBinaryFlight.h
  eoBitParticle.h
  eoBreed.h
  eoCellularEasyEA.h
  eoCloneOps.h
  eoCombinedContinue.h
  eoCombinedInit.h
  eoConstrictedVariableWeightVelocity.h
  eoConstrictedVelocity.h
  eoContinue.h
  eoCounter.h
  eoCtrlCContinue.h
  eoDetSelect.h
  eoDetTournamentSelect.h
  eoDistribUpdater.h
  eoDistribution.h
  eoDualFitness.h
  eoEasyEA.h
  eoEasyPSO.h
  eoEDA.h
  eoEvalContinue.h
  eoEvalCounterThrowException.h
  eoEvalDump.h
  eoEvalFuncCounterBounder.h
  eoEvalFuncCounter.h
  eoEvalFunc.h
  eoEvalFuncPtr.h
  eoEvalKeepBest.h
  eoEvalTimeThrowException.h
  eoEvalUserTimeThrowException.h
  eoExceptions.h
  eoExtendedVelocity.h
  eoFactory.h
  eoFitContinue.h
  eoFitnessScalingSelect.h
  eoFixedInertiaWeightedVelocity.h
  eoFlight.h
  eoFlOrBinOp.h
  eoFlOrMonOp.h
  eoFlOrQuadOp.h
  eoFunctor.h
  eoFunctorStore.h
  eoG3Replacement.h
  eoGaussRealWeightUp.h
  eoGenContinue.h
  eoGeneralBreeder.h
  eoGenOp.h
  #eo.h
  EO.h
  eoInit.h
  eoInitializer.h
  eoIntegerVelocity.h
  eoInt.h
  eoInvalidateOps.h
  eoInvertedContinue.h
  eoLinearDecreasingWeightUp.h
  eoLinearFitScaling.h
  eoLinearTopology.h
  eoMerge.h
  eoMergeReduce.h
  eoMGGReplacement.h
  eoNDSorting.h
  eoNeighborhood.h
  eoObject.h
  eoOneToOneBreeder.h
  eoOpContainer.h
  eoOp.h
  eoOpSelMason.h
  eoOrderXover.h
  eoParticleBestInit.h
  eoParticleFullInitializer.h
  eoPerf2Worth.h
  eoPeriodicContinue.h
  eoPersistent.h
  eoPopEvalFunc.h
  eoPop.h
  eoPopulator.h
  eoPrintable.h
  eoPropGAGenOp.h
  eoProportionalCombinedOp.h
  eoProportionalSelect.h
  eoPSO.h
  eoRandomRealWeightUp.h
  eoRandomSelect.h
  eoRanking.h
  eoRankingSelect.h
  eoRankMuSelect.h
  eoRealBoundModifier.h
  eoRealParticle.h
  eoReduce.h
  eoReduceMerge.h
  eoReduceMergeReduce.h
  eoReduceSplit.h
  eoReplacement.h
  eoRingTopology.h
  eoScalarFitnessAssembled.h
  eoScalarFitness.h
  eoSecondsElapsedContinue.h
  eoSelectFactory.h
  eoSelectFromWorth.h
  eoSelect.h
  eoSelectMany.h
  eoSelectNumber.h
  eoSelectOne.h
  eoSelectPerc.h
  eoSequentialSelect.h
  eoSGAGenOp.h
  eoSGA.h
  eoSGATransform.h
  eoSharing.h
  eoSharingSelect.h
  eoShiftMutation.h
  eoSigBinaryFlight.h
  eoSIGContinue.h
  eoSimpleEDA.h
  eoSocialNeighborhood.h
  eoStandardFlight.h
  eoStandardVelocity.h
  eoStarTopology.h
  eoSteadyFitContinue.h
  eoSTLFunctor.h
  eoStochasticUniversalSelect.h
  eoStochTournamentSelect.h
  eoSurviveAndDie.h
  eoSwapMutation.h
  eoSyncEasyPSO.h
  eoTimeContinue.h
  eoTopology.h
  eoTransform.h
  eoTruncatedSelectMany.h
  eoTruncatedSelectOne.h
  eoTruncSelect.h
  eoTwoOptMutation.h
  eoVariableInertiaWeightedVelocity.h
  eoVariableLengthCrossover.h
  eoVariableLengthMutation.h
  eoVector.h
  eoVectorParticle.h
  eoVelocity.h
  eoVelocityInit.h
  eoWeightUpdater.h
  es.h
  ga.h
  PO.h 
)

install(FILES eo.h DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG} COMPONENT headers)
install(FILES ${HDRS} DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/eo COMPONENT headers)
install(DIRECTORY do es ga gp other utils
        DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/eo
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h" PATTERN "checkpointing" PATTERN external_eo
        )

######################################################################################
### 4) Where must cmake go now ?
######################################################################################

add_subdirectory(es)
add_subdirectory(ga)
add_subdirectory(utils)
#add_subdirectory(serial)

if(ENABLE_PYEO)
  add_subdirectory(pyeo)
endif(ENABLE_PYEO)

######################################################################################
