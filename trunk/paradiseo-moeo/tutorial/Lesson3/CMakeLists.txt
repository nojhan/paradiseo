
######################################################################################
### 0) Copy the FlowShop.param file in the build path for an easy use.
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson3/FlowShopEA.param)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson3/FlowShopEA.param
      ${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson3)
######################################################################################



######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${ParadisEO-MOEO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${FLOWSHOP_SRC_DIR})

######################################################################################



######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-MOEO_BINARY_DIR}/lib ${FLOWSHOP_BINARY_DIR}/lib)

######################################################################################



######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

ADD_EXECUTABLE(FlowShopEA FlowShopEA.cpp) 
ADD_DEPENDENCIES(FlowShopEA flowshop moeo)
######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(FLOWSHOPEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(FlowShopEA PROPERTIES VERSION "${FLOWSHOPEA_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################


TARGET_LINK_LIBRARIES(FlowShopEA moeo flowshop eo eoutils)

######################################################################################
