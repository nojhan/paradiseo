
######################################################################################
### 0) Copy the FlowShop.param file in the build path for an easy use.
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson1/FlowShopEA.param)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-MOEO_SOURCE_DIR}/tutorial/Lesson1/FlowShopEA.param
      ${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson1)
######################################################################################



######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR})
INCLUDE_DIRECTORIES(${EO_SRC_DIR}/utils)
INCLUDE_DIRECTORIES(${MOEO_SRC_DIR})
INCLUDE_DIRECTORIES(${FLOWSHOP_SRC_DIR})

######################################################################################



######################################################################################
### 2) Specify where CMake can find the libraries (mandatory: before 3) )
######################################################################################

# --> UNIX 
IF(UNIX)  
  LINK_DIRECTORIES(${EO_SRC_DIR} ${EO_SRC_DIR}/utils ${ParadisEO-MOEO_BINARY_DIR}/lib ${FLOWSHOP_BINARY_DIR}/lib)

# --> WIN
ELSEIF(WIN32)  
  # "CMAKE_BUILD_TYPE" supposed to be given on the command line, default=Debug
  IF (NOT DEFINED CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Debug)
  ENDIF (NOT DEFINED CMAKE_BUILD_TYPE)
 
  IF (CMAKE_BUILD_TYPE STREQUAL Debug)
  	LINK_DIRECTORIES(${FLOWSHOP_BINARY_DIR}\\lib\\debug)
  	LINK_DIRECTORIES(${ParadisEO-MOEO_BINARY_DIR}\\lib\\debug)
  	LINK_DIRECTORIES(${EO_LIB_DIR}\\debug)
  ELSEIF(CMAKE_BUILD_TYPE STREQUAL Release)
  	LINK_DIRECTORIES(${FLOWSHOP_BINARY_DIR}\\lib\\release)
  	LINK_DIRECTORIES(${ParadisEO-MOEO_BINARY_DIR}\\lib\\release)
  	LINK_DIRECTORIES(${EO_LIB_DIR}\\release)  
  ENDIF(CMAKE_BUILD_TYPE STREQUAL Debug)

ENDIF(UNIX)

######################################################################################



######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

# no matter what is the OS, hopefully
ADD_EXECUTABLE(FlowShopEA FlowShopEA.cpp) 

ADD_DEPENDENCIES(FlowShopEA moeo)
ADD_DEPENDENCIES(FlowShopEA flowshop)
######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(FLOWSHOPEA_VERSION "1.0.beta")
SET_TARGET_PROPERTIES(FlowShopEA PROPERTIES VERSION "${FLOWSHOPEA_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

# --> UNIX 
IF(UNIX) 
  TARGET_LINK_LIBRARIES(FlowShopEA moeo)
  TARGET_LINK_LIBRARIES(FlowShopEA flowshop) 
  TARGET_LINK_LIBRARIES(FlowShopEA eo)
  TARGET_LINK_LIBRARIES(FlowShopEA eoutils)

# --> WIN
ELSEIF(WIN32)  
  # "CMAKE_BUILD_TYPE" supposed to be given on the command line, default=Debug
  IF (NOT DEFINED CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Debug)
  ENDIF (NOT DEFINED CMAKE_BUILD_TYPE)
 
  IF (CMAKE_BUILD_TYPE STREQUAL Debug)
  	TARGET_LINK_LIBRARIES(FlowShopEA moeo)
  	TARGET_LINK_LIBRARIES(FlowShopEA flowshop)
    TARGET_LINK_LIBRARIES(FlowShopEA eod)
    TARGET_LINK_LIBRARIES(FlowShopEA eoutilsd)    
  ELSEIF(CMAKE_BUILD_TYPE STREQUAL Release)
  	TARGET_LINK_LIBRARIES(FlowShopEA moeo)
  	TARGET_LINK_LIBRARIES(FlowShopEA flowshop)
    TARGET_LINK_LIBRARIES(FlowShopEA eo)
    TARGET_LINK_LIBRARIES(FlowShopEA eoutils)   
  ELSE(CMAKE_BUILD_TYPE STREQUAL Debug)
    MESSAGE(FATAL_ERROR " Cmake build type can only be \"Debug\" or \"Release\" ")
  ENDIF(CMAKE_BUILD_TYPE STREQUAL Debug)

ENDIF(UNIX)
######################################################################################



######################################################################################
### 6) Windows advanced config - especially for Microsoft Visual Studio 8
######################################################################################

  IF(CMAKE_CXX_COMPILER MATCHES cl)
   IF(NOT WITH_SHARED_LIBS)
     IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
       SET(CMAKE_CXX_FLAGS "/nologo /W3 /Gy")
       SET(CMAKE_CXX_FLAGS_DEBUG "/MTd /Z7 /Od")
       SET(CMAKE_CXX_FLAGS_RELEASE "/MT /O2")
       SET(CMAKE_CXX_FLAGS_MINSIZEREL "/MT /O2")
       SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MTd /Z7 /Od")
       SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
       
     ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
   ENDIF(NOT WITH_SHARED_LIBS)
  ENDIF(CMAKE_CXX_COMPILER MATCHES cl)
######################################################################################

