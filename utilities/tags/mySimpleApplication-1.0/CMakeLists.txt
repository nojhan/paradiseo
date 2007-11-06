
######################################################################################
### 0) Include the install configuration file where are defined the main variables
######################################################################################

INCLUDE(${CMAKE_SOURCE_DIR}/install.cmake)

######################################################################################


######################################################################################
### 1) Set your application properties
######################################################################################

# Define your project name
PROJECT(mySimpleApplication)

######################################################################################


######################################################################################
### 2) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src 
                    ${PARADISEO_MO_SRC_DIR}/src  
                    ${PARADISEO_MOEO_SRC_DIR}/src 
                    ${PARADISEO_PEO_SRC_DIR}/src)

######################################################################################


######################################################################################
### 3) Specify where CMake can find the libraries
######################################################################################

# Link with the EO library directory, only if you need it
LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib)

# Link with the MOEO library directory, only if you need it
LINK_DIRECTORIES(${PARADISEO_MOEO_BIN_DIR}/lib)

# Link with the PEO library directory, only if you need it
LINK_DIRECTORIES(${PARADISEO_PEO_BIN_DIR}/lib)

######################################################################################



######################################################################################
### 4) Define your target: just an executable here
######################################################################################

ADD_EXECUTABLE(mySimpleApplication mySimpleApplication.cpp) 

######################################################################################


######################################################################################
### 5) Link the librairies for your executable
######################################################################################

# Only if you need to link libraries
TARGET_LINK_LIBRARIES(mySimpleApplication eo eoutils) 

######################################################################################






