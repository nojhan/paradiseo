######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src/utils)
INCLUDE_DIRECTORIES(${PARADISEO_MOEO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${SOURCES_DIR}/flowshop/src)
######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib)

LINK_DIRECTORIES(${PARADISEO_MOEO_BIN_DIR}/lib)

LINK_DIRECTORIES(${INSTALL_DIR}/flowshop/lib)

######################################################################################


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${SOURCES_DIR}/MOEO_lesson/param )

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${SOURCES_DIR}/MOEO_lesson/param
      ${INSTALL_DIR}/MOEO_lesson)

ADD_EXECUTABLE(FlowShopEA FlowShopEA.cpp) 
ADD_DEPENDENCIES(FlowShopEA flowshop)

######################################################################################


######################################################################################
### 4) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(FlowShopEA flowshop eo eoutils moeo)

######################################################################################

