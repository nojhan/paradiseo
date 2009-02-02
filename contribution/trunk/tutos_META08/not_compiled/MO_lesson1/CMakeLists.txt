######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src/utils)
INCLUDE_DIRECTORIES(${PARADISEO_MO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${SOURCES_DIR}/tsp/src)
######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib ${INSTALL_DIR}/tsp/lib)

######################################################################################


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${SOURCES_DIR}/MO_lesson1/param )

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${SOURCES_DIR}/MO_lesson1/param
      ${INSTALL_DIR}/MO_lesson1)

ADD_EXECUTABLE(hill_climbing hill_climbing.cpp) 

ADD_DEPENDENCIES(hill_climbing tsp)
######################################################################################


######################################################################################
### 4) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(hill_climbing tsp eo eoutils) 

######################################################################################


