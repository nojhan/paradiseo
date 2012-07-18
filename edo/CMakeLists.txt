############################################################################
##########
### 1) If you want to set your own variables in install.cmake and avoid the cmd line
######################################################################################

INCLUDE(install.cmake OPTIONAL)

######################################################################################

######################################################################################
### 2) Project properties
######################################################################################

# Checks cmake version compatibility
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(EDO)

SET(PROJECT_VERSION_MAJOR 0)
SET(PROJECT_VERSION_MINOR 1)
SET(PROJECT_VERSION_PATCH 0)
SET(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

######################################################################################


######################################################################################
### 3) Include useful features
######################################################################################

# include useful features for cmake
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/cmake/modules)


INCLUDE(FindDoxygen)
INCLUDE(FindPkgConfig)

IF( WITH_BOOST AND WITH_EIGEN )
    MESSAGE( "ERROR: You have to choose between Boost:ublas and Eigen, you cannot compile with both libraries" )
    SET(IS_FATAL 1)
ELSEIF( NOT WITH_BOOST AND NOT WITH_EIGEN )
    #MESSAGE( "WARNING: Boost:ublas and Eigen are both deactivated, some features may lack." )
    # FIXME ideally, we would have a minimal implementation with STL vectors…
    MESSAGE( "FIXME: Boost:ublas and Eigen are both deactivated, too much features will lack, you should choose one." )
    SET(IS_FATAL 1)
ENDIF()

IF(WITH_BOOST)
    FIND_PACKAGE(Boost 1.33.0)
    IF( Boost_FOUND )
        INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
        ADD_DEFINITIONS( -DWITH_BOOST )
    ELSE()
        MESSAGE( "ERROR: You asked for Boost:ublas but it has nost been found." )
        SET(IS_FATAL 1)
    ENDIF()
ELSEIF( WITH_EIGEN )
    # FIXME FindEigen3.cmake does not work
    #find_package(Eigen3)
    #include_directories(EIGEN3_INCLUDE_DIR)
    SET( EIGEN3_FOUND 1)
    SET( EIGEN3_INCLUDE_DIR "/usr/include/eigen3/" )

    IF( EIGEN3_FOUND )
        INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIR} )
        ADD_DEFINITIONS( -DWITH_EIGEN )
    ELSE()
        MESSAGE( "ERROR: You asked for Eigen but it has nost been found." )
        SET(IS_FATAL 1)
    ENDIF()
ENDIF()

FIND_PACKAGE(EO)

INCLUDE_DIRECTORIES(
  ${EO_INCLUDE_DIRS}
  ${MO_INCLUDE_DIRS}
)

LINK_DIRECTORIES(
  ${EO_LIBRARY_DIRS}
)

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

# Empty source files, because we want to build a library
SET(SAMPLE_SRCS)

######################################################################################


######################################################################################
### 7) Now where we go ?
######################################################################################

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(application)
#ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(doc)

######################################################################################



######################################################################################
### 8) Create executable, link libraries and prepare target
######################################################################################

SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})

ADD_LIBRARY(edo STATIC ${SAMPLE_SRCS})
INSTALL(TARGETS edo ARCHIVE DESTINATION lib COMPONENT libraries)

######################################################################################


######################################################################################
### 9) Install pkg-config config file for EO
######################################################################################

INSTALL(FILES edo.pc DESTINATION lib/pkgconfig COMPONENT headers)

######################################################################################


######################################################################################
### 10) Include packaging
######################################################################################

INCLUDE(Packaging.cmake)

######################################################################################
