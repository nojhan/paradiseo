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

find_package(Boost 1.33.0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/application/common)
include_directories(${EO_SRC_DIR}/src)
include_directories(${EDO_SRC_DIR}/src)

set(SOURCES
    #t-cholesky
  t-variance
  t-edoEstimatorNormalMulti
  t-mean-distance
  t-bounderno
  t-uniform
  t-continue
  t-dispatcher-round
  t-repairer-modulo
  )

foreach(current ${SOURCES})
  add_executable(${current} ${current}.cpp)
  add_test(${current} ${current})
  target_link_libraries(${current} eo eoutils edoutils ${Boost_LIBRARIES})
  install(TARGETS ${current} RUNTIME DESTINATION share/edo/test COMPONENT test)
endforeach()

######################################################################################
