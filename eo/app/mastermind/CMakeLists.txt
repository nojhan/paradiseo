######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Specify where CMake can find the libraries (mandatory: before 3) )
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/${LIB})

######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

SET (MASTERMIND_SOURCES mastermind.cpp)

# no matter what is the OS, hopefully
ADD_EXECUTABLE(mastermind ${MASTERMIND_SOURCES})

ADD_DEPENDENCIES(mastermind eo eoutils)

######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(MASTERMIND_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(mastermind PROPERTIES VERSION "${MASTERMIND_VERSION}")

######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(mastermind eo eoutils)

######################################################################################
