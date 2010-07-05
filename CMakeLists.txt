######################################################################################
### 1) Set the application properties
######################################################################################

# Checks cmake version compatibility
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(DO)

SET(PROJECT_VERSION_MAJOR 1)
SET(PROJECT_VERSION_MINOR 0)
SET(PROJECT_VERSION_PATCH 0)
SET(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

######################################################################################


######################################################################################
### 2) Prepare some useful variables
######################################################################################

SET(DO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
SET(DO_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}")

######################################################################################


######################################################################################
### 3) Include useful features
######################################################################################

INCLUDE(FindDoxygen)

INCLUDE(FindPkgConfig)

PKG_CHECK_MODULES(EO eo REQUIRED)

INCLUDE_DIRECTORIES(${EO_INCLUDE_DIR})
LINK_DIRECTORIES(${EO_LIBRARY_DIRS})

######################################################################################


######################################################################################
### 4) Include header files path
######################################################################################

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

######################################################################################


######################################################################################
### 5) Set compiler definitions
######################################################################################

IF(UNIX)
  # enable warnings
  ADD_DEFINITIONS( -Wall -W -Wextra )
  # ADD_DEFINITIONS( -Weffc++)
  # ADD_DEFINITIONS( -g3 )
ENDIF()

######################################################################################


######################################################################################
### 6) Prepare some variables for CMAKE usage
######################################################################################

SET(SAMPLE_SRCS)

######################################################################################


######################################################################################
### 7) Now where we go ?
######################################################################################

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(application)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(doc)

######################################################################################


######################################################################################
### 8) Create executable, link libraries and prepare target
######################################################################################

SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})

ADD_LIBRARY(${PROJECT_NAME} STATIC ${SAMPLE_SRCS})
#ADD_LIBRARY(${PROJECT_NAME} SHARED ${SAMPLE_SRCS})

# INSTALL(
#   TARGETS ${LIBRARY_OUTPUT_PATH}/lib${PROJECT_NAME}.a
#   DESTINATION lib
#   COMPONENT libraries
#   )

######################################################################################
### 9) Include packaging
######################################################################################

INCLUDE(Packaging.cmake)

######################################################################################
