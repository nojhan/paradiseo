######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src/ga)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src/utils)

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
ADD_EXECUTABLE(SecondBitEA SecondBitEA.cpp)
ADD_EXECUTABLE(SecondRealEA SecondRealEA.cpp)
ADD_EXECUTABLE(exercise3.1 exercise3.1.cpp)

ADD_DEPENDENCIES(SecondBitEA ga eoutils eo)
ADD_DEPENDENCIES(SecondRealEA ga eoutils eo)
ADD_DEPENDENCIES(exercise3.1 ga eoutils eo)

######################################################################################
### 4) Optionnal
######################################################################################

SET(SECONDBITEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(SecondBitEA PROPERTIES VERSION "${SECONDBITEA_VERSION}")

SET(SECONDREALEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(SecondRealEA PROPERTIES VERSION "${SECONDREALEA_VERSION}")

SET(EXERCICE31_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(exercise3.1 PROPERTIES VERSION "${EXERCICE31_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

TARGET_LINK_LIBRARIES(SecondBitEA ga eoutils eo)
TARGET_LINK_LIBRARIES(SecondRealEA ga eoutils eo)
TARGET_LINK_LIBRARIES(exercise3.1 ga eoutils eo)

######################################################################################
### 6) Configure project installation paths
######################################################################################

INSTALL(TARGETS SecondBitEA RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson3 COMPONENT examples)
INSTALL(TARGETS SecondRealEA RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson3 COMPONENT examples)
INSTALL(TARGETS exercise3.1 RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson3 COMPONENT examples)

######################################################################################
