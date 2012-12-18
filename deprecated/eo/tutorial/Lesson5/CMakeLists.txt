######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

IF(NOT WIN32 OR CYGWIN)
  LINK_DIRECTORIES(${EO_BINARY_DIR}/lib)
ENDIF(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
IF(WIN32 AND NOT CYGWIN)
  	LINK_DIRECTORIES(${EO_BINARY_DIR}\\lib\\${CMAKE_BUILD_TYPE})
ENDIF(WIN32 AND NOT CYGWIN)

######################################################################################
### 3) Define your targets
######################################################################################

# no matter what is the OS, hopefully
ADD_EXECUTABLE(OneMaxEA OneMaxEA.cpp)
ADD_EXECUTABLE(OneMaxLibEA OneMaxLibEA.cpp make_OneMax.cpp)

ADD_DEPENDENCIES(OneMaxEA es ga eo eoutils)
ADD_DEPENDENCIES(OneMaxLibEA es ga eo eoutils)

######################################################################################
### 4) Optionnal
######################################################################################

SET(ONEMAXEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(OneMaxEA PROPERTIES VERSION "${ONEMAXEA_VERSION}")

SET(ONEMAXLIBEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(OneMaxLibEA PROPERTIES VERSION "${ONEMAXLIBEA_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

TARGET_LINK_LIBRARIES(OneMaxEA es ga eo eoutils)
TARGET_LINK_LIBRARIES(OneMaxLibEA es ga eo eoutils)

######################################################################################
### 6) Configure project installation paths
######################################################################################

INSTALL(TARGETS OneMaxEA RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson5 COMPONENT examples)
INSTALL(TARGETS OneMaxLibEA RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson5 COMPONENT examples)

######################################################################################
