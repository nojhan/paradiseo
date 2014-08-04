###############################################################################
##
## CMakeLists file for unit test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

#include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#include_directories(${CMAKE_SOURCE_DIR}/application/common)
include_directories(${EDO_TUTORIAL_DIR}/common)
#include_directories(${EO_SRC_DIR}/src)
#include_directories(${EDO_SRC_DIR}/src)

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

if(EDO_USE_LIB STREQUAL "Eigen3")
  find_package(Eigen3)
  include_directories( ${EIGEN3_INCLUDE_DIR} )
  add_definitions( -DWITH_EIGEN )
elseif(EDO_USE_LIB STREQUAL "uBLAS")
  find_package(Boost)
  include_directories( ${Boost_INCLUDE_DIRS} )
  add_definitions( -DWITH_BOOST )
else()
  message(FATAL_ERROR  "\n\nYou must set EDO_USE_LIB to either 'uBLAS' or 'Eigen3'.\n" )
endif()

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

set(SOURCES
    #t-cholesky
  t-variance
  t-edoEstimatorNormalMulti
  t-mean-distance
  t-bounderno
  t-uniform
  t-continue
#  t-dispatcher-round
  t-repairer-modulo
  t-binomial
  t-binomialmulti
  )

foreach(current ${SOURCES})
  add_executable(${current} ${current}.cpp)
  add_test(${current} ${current})
  target_link_libraries(${current} eo eoutils edoutils)
  install(TARGETS ${current} RUNTIME DESTINATION share/edo/test COMPONENT test)
endforeach()

######################################################################################
