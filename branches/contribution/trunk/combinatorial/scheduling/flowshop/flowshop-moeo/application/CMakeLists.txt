######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src
                    ${PARADISEO_MOEO_SRC_DIR}/src
                    ${flow-shop_SOURCE_DIR}/src)

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib)
LINK_DIRECTORIES(${PARADISEO_MOEO_BIN_DIR}/lib)
LINK_DIRECTORIES(${flow-shop_BINARY_DIR}/lib)

######################################################################################


######################################################################################
### 3) Define your target: just an executable here
######################################################################################

ADD_EXECUTABLE(FlowShopMOEA FlowShopMOEA.cpp)
ADD_DEPENDENCIES(FlowShopMOEA lflowshop)

######################################################################################


######################################################################################
### 4) Link the librairies for your executable
######################################################################################

# Only if you need to link libraries
TARGET_LINK_LIBRARIES(FlowShopMOEA moeo eo eoutils lflowshop)

######################################################################################


######################################################################################
### 5) Copy the instances and the "param" file in the build path for an easy use.
### 
### --> run the "make install" target to copy the parameter file / instances
###	in the directory where you build the application
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${flow-shop_SOURCE_DIR}/application/FlowShopMOEA.param
				  ${flow-shop_SOURCE_DIR}/application/instances)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${flow-shop_SOURCE_DIR}/application/FlowShopMOEA.param
      ${flow-shop_BINARY_DIR}/application)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory
      ${flow-shop_SOURCE_DIR}/application/instances
      ${flow-shop_BINARY_DIR}/application/instances)
######################################################################################
