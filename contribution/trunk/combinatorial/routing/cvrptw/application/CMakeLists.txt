

######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src
                    ${CVRP-TW_SOURCE_DIR}/src)

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

# Link with the EO library directory
LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib)

######################################################################################



######################################################################################
### 3) Define your target: just an executable here
######################################################################################

ADD_EXECUTABLE(CVRP-TW CVRP-TW.cpp)

######################################################################################


######################################################################################
### 4) Link the librairies for your executable
######################################################################################

# Only if you need to link libraries
TARGET_LINK_LIBRARIES(CVRP-TW eo eoutils)

######################################################################################


######################################################################################
### 5) Copy the instances and the "param" file in the build path for an easy use.
### 
### --> run the "make install" target to copy the parameter file / instances
###	in the directory where you build the application
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${CVRP-TW_SOURCE_DIR}/application/CVRP-TW.params
				  ${CVRP-TW_SOURCE_DIR}/application/instances)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${CVRP-TW_SOURCE_DIR}/application/CVRP-TW.params
      ${CVRP-TW_BINARY_DIR}/application)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory
      ${CVRP-TW_SOURCE_DIR}/application/instances
      ${CVRP-TW_BINARY_DIR}/application)
######################################################################################


