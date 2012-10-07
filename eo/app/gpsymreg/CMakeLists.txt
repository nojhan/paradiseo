######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) specify where cmake can find the libraries (mandatory: before 3) )
######################################################################################

link_directories(${EO_BIN_DIR}/lib)

######################################################################################
### 3) define your target(s): just an executable here
######################################################################################

set (GPSYMREG_SOURCES main.cpp)

# no matter what is the OS, hopefully
add_executable(gpsymreg ${GPSYMREG_SOURCES})

add_dependencies(gpsymreg eo eoutils)

######################################################################################
### 4) optionnal: define your target(s)'s version: no effect for windows
######################################################################################

set(GPSYMREG_VERSION ${GLOBAL_VERSION})
set_target_properties(gpsymreg PROPERTIES VERSION "${GPSYMREG_VERSION}")

######################################################################################
### 5) link the librairies for your target(s)
######################################################################################

target_link_libraries(gpsymreg eo eoutils)

######################################################################################
