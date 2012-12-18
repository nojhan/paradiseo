######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Specify where CMake can find the libraries (mandatory: before 3) )
######################################################################################

LINK_DIRECTORIES(${EO_BINARY_DIR}/lib)

######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

SET (GPSYMREG_SOURCES main.cpp)

# no matter what is the OS, hopefully
ADD_EXECUTABLE(gpsymreg ${GPSYMREG_SOURCES})

ADD_DEPENDENCIES(gpsymreg eo eoutils)

######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(GPSYMREG_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(gpsymreg PROPERTIES VERSION "${GPSYMREG_VERSION}")

######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(gpsymreg eo eoutils)

######################################################################################
