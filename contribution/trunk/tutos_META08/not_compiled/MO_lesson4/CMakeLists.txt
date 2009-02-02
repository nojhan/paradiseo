

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

ADD_CUSTOM_TARGET(install DEPENDS ${SOURCES_DIR}/MO_lesson4/param )

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${SOURCES_DIR}/MO_lesson4/param
      ${INSTALL_DIR}/MO_lesson4)

ADD_EXECUTABLE(iterated_local_search iterated_local_search.cpp) 

ADD_DEPENDENCIES(iterated_local_search tsp)

######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

#SET(ITERATEDLOCALSEARCH_VERSION ${GLOBAL_VERSION})
#SET_TARGET_PROPERTIES(iterated_local_search PROPERTIES VERSION "${ITERATEDLOCALSEARCH_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(iterated_local_search tsp eo eoutils) 

######################################################################################
