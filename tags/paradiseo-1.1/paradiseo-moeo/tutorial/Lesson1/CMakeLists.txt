
######################################################################################
### 0) Copy the Sch1.param file in the build path for an easy use.
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson1/Sch1.param)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson1/Sch1.param
      ${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson1)
######################################################################################



######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SRC_DIR}/utils)
INCLUDE_DIRECTORIES(${ParadisEO-MOEO_SOURCE_DIR}/src)
######################################################################################



######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################
 
LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-MOEO_BINARY_DIR}/lib)

######################################################################################



######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

ADD_EXECUTABLE(Sch1 Sch1.cpp)
ADD_DEPENDENCIES(Sch1 moeo)
######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(SCH1_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(Sch1 PROPERTIES VERSION "${SCH1_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################


TARGET_LINK_LIBRARIES(Sch1 moeo eo eoutils)

######################################################################################
