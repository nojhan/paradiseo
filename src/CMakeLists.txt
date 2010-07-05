######################################################################################
### 1) Include useful packages
######################################################################################

FIND_PACKAGE(Boost 1.33.0 REQUIRED)

######################################################################################


######################################################################################
### 2) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(
  ${PARADISEO_EO_SRC_DIR}/src
  ${PARADISEO_MO_SRC_DIR}/src
  ${PARADISEO_MOEO_SRC_DIR}/src
  ${PARADISEO_PEO_SRC_DIR}/src
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/lib/eometah
  ${CMAKE_SOURCE_DIR}/lib/nk-landscapes
  ${Boost_INCLUDE_DIRS}
  )

######################################################################################


######################################################################################
### 3) Define your target(s)
######################################################################################

SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

FILE(GLOB SOURCES *.cpp)

ADD_LIBRARY(${PROJECT_NAME} STATIC ${SOURCES})

# INSTALL(
#   TARGETS ${LIBRARY_OUTPUT_PATH}/lib${PROJECT_NAME}.a
#   DESTINATION lib
#   COMPONENT libraries
#   )

######################################################################################
