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
ADD_EXECUTABLE(FirstBitEA FirstBitEA.cpp)
ADD_EXECUTABLE(FirstRealEA FirstRealEA.cpp)
ADD_EXECUTABLE(exercise2.3 exercise2.3.cpp)

######################################################################################
### 4) Optionnal
######################################################################################

SET(FIRSTBITEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(FirstBitEA PROPERTIES VERSION "${FIRSTBITEA_VERSION}")

SET(FIRSTREALEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(FirstRealEA PROPERTIES VERSION "${FIRSTREALEA_VERSION}")

SET(EXERCICE23_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(exercise2.3 PROPERTIES VERSION "${EXERCICE23_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

TARGET_LINK_LIBRARIES(FirstBitEA ga eo eoutils)
TARGET_LINK_LIBRARIES(FirstRealEA ga eo eoutils)
TARGET_LINK_LIBRARIES(exercise2.3 ga eo eoutils)

######################################################################################
### 6) Configure project installation paths
######################################################################################

INSTALL(TARGETS FirstBitEA RUNTIME DESTINATION local/share/${PROJECT_TAG}/eo/examples/Lesson2 COMPONENT examples)
INSTALL(TARGETS FirstRealEA RUNTIME DESTINATION local/share/${PROJECT_TAG}/eo/examples/Lesson2 COMPONENT examples)
INSTALL(TARGETS exercise2.3 RUNTIME DESTINATION local/share/${PROJECT_TAG}/eo/examples/Lesson2 COMPONENT examples)

######################################################################################
