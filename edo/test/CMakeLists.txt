###############################################################################
##
## CMakeLists file for unit test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

######################################################################################


######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

FIND_PACKAGE(Boost 1.33.0)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/application/common)

SET(SOURCES
  t-edoEstimatorNormalMulti
  t-mean-distance
  t-bounderno
  t-uniform
  t-continue
  t-dispatcher-round
  )

FOREACH(current ${SOURCES})
  ADD_EXECUTABLE(${current} ${current}.cpp)
  ADD_TEST(${current} ${current})
  TARGET_LINK_LIBRARIES(${current} edo edoutils ${EO_LIBRARIES} ${MO_LIBRARIES} ${Boost_LIBRARIES})
  INSTALL(TARGETS ${current} RUNTIME DESTINATION share/edo/test COMPONENT test)
ENDFOREACH()

######################################################################################
