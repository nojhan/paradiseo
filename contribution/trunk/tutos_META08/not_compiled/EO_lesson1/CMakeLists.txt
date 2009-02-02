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


#IF(NOT WIN32 OR CYGWIN)  
  LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib ${INSTALL_DIR}/tsp/lib)
#ENDIF(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
#IF(WIN32 AND NOT CYGWIN)  
#  	LINK_DIRECTORIES(${EO_BINARY_DIR}\\lib\\${CMAKE_BUILD_TYPE${INSTALL_DIR}\\tsp\\lib\\${CMAKE_BUILD_TYPE})
#ENDIF(WIN32 AND NOT CYGWIN) 

######################################################################################


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${SOURCES_DIR}/EO_lesson1/param )

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${SOURCES_DIR}/EO_lesson1/param
      ${INSTALL_DIR}/EO_lesson1)

ADD_EXECUTABLE(tsp_EA tsp.cpp) 

ADD_DEPENDENCIES(tsp_EA tsp)
######################################################################################


######################################################################################
### 4) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(tsp_EA tsp eo eoutils) 

######################################################################################


