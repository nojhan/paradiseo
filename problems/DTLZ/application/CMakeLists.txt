######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src
                    ${PARADISEO_MO_SRC_DIR}/src
                    ${PARADISEO_MOEO_SRC_DIR}/src
                    ${DTLZ_SOURCE_DIR}/src)

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib)
LINK_DIRECTORIES(${PARADISEO_MOEO_BIN_DIR}/lib)
LINK_DIRECTORIES(${DTLZ_BINARY_DIR}/lib)
######################################################################################


######################################################################################
### 3) Define your target: just an executable here
######################################################################################

ADD_EXECUTABLE(DTLZ_SPEA2 DTLZ_SPEA2.cpp)
ADD_EXECUTABLE(DTLZ_NSGAII DTLZ_NSGAII.cpp)
ADD_EXECUTABLE(DTLZ_IBEA DTLZ_IBEA.cpp)
ADD_DEPENDENCIES(DTLZ_SPEA2 lDTLZ)
ADD_DEPENDENCIES(DTLZ_NSGAII lDTLZ)
ADD_DEPENDENCIES(DTLZ_IBEA lDTLZ)

######################################################################################


######################################################################################
### 4) Link the librairies for your executable
######################################################################################

# Only if you need to link libraries
TARGET_LINK_LIBRARIES(DTLZ_SPEA2 moeo eo eoutils lDTLZ)
TARGET_LINK_LIBRARIES(DTLZ_NSGAII moeo eo eoutils lDTLZ)
TARGET_LINK_LIBRARIES(DTLZ_IBEA moeo eo eoutils lDTLZ)

######################################################################################


######################################################################################
### 5) Copy the instances and the "param" file in the build path for an easy use.
### 
### --> run the "make install" target to copy the parameter file / instances
###	in the directory where you build the application
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${DTLZ_SOURCE_DIR}/application/IBEA.param
				  ${DTLZ_SOURCE_DIR}/application/NSGAII.param
				  ${DTLZ_SOURCE_DIR}/application/SPEA2.param
				  ${DTLZ_SOURCE_DIR}/application)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${DTLZ_SOURCE_DIR}/application/IBEA.param
      ${DTLZ_BINARY_DIR}/application
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${DTLZ_SOURCE_DIR}/application/NSGAII.param
      ${DTLZ_BINARY_DIR}/application
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${DTLZ_SOURCE_DIR}/application/SPEA2.param
      ${DTLZ_BINARY_DIR}/application)

######################################################################################
