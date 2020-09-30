###############################################################################
##
## CMakeLists file for eo/test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${EO_SRC_DIR}/contrib)
include_directories(${EO_SRC_DIR}/contrib/MGE)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

link_directories(${EO_BIN_DIR}/lib)

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

set (TEST_LIST
  t-eofitness
  t-eoRandom
  t-eobin
  t-eoVirus
  t-MGE
  t-MGE1bit
  t-MGE-control
  t-eoStateAndParser
  t-eoCheckpointing
  t-eoSSGA
  t-eoExternalEO
  t-eoSymreg
  t-eo
  t-eoReplacement
  t-eoSelect
  t-eoGenOp
  t-eoGA
  t-eoReal
  t-eoVector
  t-eoESAll
  # t-eoPBIL # deprecated, see EDO module.
  t-eoFitnessAssembled
  t-eoFitnessAssembledEA
  t-eoRoulette
  t-eoSharing
  t-eoCMAES
  t-eoSecondsElapsedContinue
  t-eoRNG
  t-eoEasyPSO
  t-eoInt
  t-eoInitPermutation
  t-eoSwapMutation
  t-eoShiftMutation
  t-eoTwoOptMutation
  t-eoRingTopology
  t-eoSyncEasyPSO
  t-eoOrderXover
  t-eoExtendedVelocity
  t-eoLogger
  #t-eoIQRStat # Temporary by-passed in order to test coverage
  t-eoParallel
  #t-openmp # does not work anymore since functions used in this test were removed from EO
  #t-eoDualFitness
  t-eoParser
  t-eoPartiallyMappedXover
  t-eoEvalCmd
  t-operator-forge
  t-forge-algo
  t-algo-forged
  t-algo-forged-search
  t-FastGA
  t-eoFastGA
  t-forge-FastGA
  t-eoFoundryFastGA
  t-eoAlgoFoundryFastGA
  )


foreach (test ${TEST_LIST})
  set ("T_${test}_SOURCES" "${test}.cpp")
endforeach (test)


if(ENABLE_MINIMAL_CMAKE_TESTING)

  set (MIN_TEST_LIST t-eoEasyPSO)
  foreach (mintest ${MIN_TEST_LIST})
    set ("T_${mintest}_SOURCES" "${mintest}.cpp")
    add_executable(${mintest} ${T_${mintest}_SOURCES})
    add_test(${mintest} ${mintest})
    target_link_libraries(${mintest} ga es cma eoutils eo)
  endforeach (mintest)

elseif(ENABLE_CMAKE_TESTING)

  foreach (test ${TEST_LIST})
    add_executable(${test} ${T_${test}_SOURCES})
    add_test(${test} ${test})
    target_link_libraries(${test} ga es cma eoutils eo)
    install(TARGETS ${test} RUNTIME DESTINATION share${INSTALL_SUB_DIR}/eo/test COMPONENT test)
  endforeach (test)

  set(RESOURCES
    boxplot.py
    boxplot_to_png.py
    boxplot_to_pdf.py
    t-openmp.py
    )

  foreach(file ${RESOURCES})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_SOURCE_DIR}/${file}
      ${CMAKE_CURRENT_BINARY_DIR}/${file}
      )
  endforeach(file)

endif(ENABLE_MINIMAL_CMAKE_TESTING)

######################################################################################
