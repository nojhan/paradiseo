CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

######################################################################################
### 0) If you want to set your variables in eo-conf.cmake and avoid the cmd line
######################################################################################

#INCLUDE(eo-conf.cmake OPTIONAL)

######################################################################################


######################################################################################
### 1) Main project config
######################################################################################

# set the project namef and other variables
PROJECT(EO)

# CMake > 2.8 is needed, because of the FindOpenMP feature
#cmake_minimum_required(VERSION 2.8)

#SET(PROJECT_VERSION_MAJOR 1)
#SET(PROJECT_VERSION_MINOR 1)
#SET(PROJECT_VERSION_PATCH 1)
SET(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}" CACHE STRING "Package version" FORCE)
SET(VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}${PROJECT_VERSION_MISC}" CACHE STRING "Global version" FORCE)
SET(GLOBAL_VERSION "${VERSION}")

SET(PACKAGE_BUGREPORT "eodev-help@sourceforge.net" CACHE STRING "Package bug report" FORCE)
SET(PACKAGE_NAME "Evolving Objects" CACHE STRING "Package name" FORCE)
SET(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}"CACHE STRING "Package string full name" FORCE)
SET(PACKAGE_TARNAME "eo" CACHE STRING "Package tar name" FORCE)

# regular expression checking
INCLUDE_REGULAR_EXPRESSION("^.*$" "^$")

# set a language for the entire project.
ENABLE_LANGUAGE(CXX)
ENABLE_LANGUAGE(C)

######################################################################################


#####################################################################################
### 2) Include required modules / configuration files
#####################################################################################

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

INCLUDE(CMakeBackwardCompatibilityCXX)

INCLUDE(FindDoxygen)

INCLUDE(FindGnuplot)

INCLUDE(CheckLibraryExists)

IF(UNIX)
  INCLUDE(ConfigureChecks.cmake)
ENDIF(UNIX)

INCLUDE(Dart OPTIONNAL)

# now create config headers
CONFIGURE_FILE(config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)

# now create config install_symlink script file
CONFIGURE_FILE(install_symlink.py.cmake ${CMAKE_CURRENT_BINARY_DIR}/install_symlink.py)

# now create PKGBUILD file for archlinux package manager
CONFIGURE_FILE(PKGBUILD.cmake ${CMAKE_CURRENT_BINARY_DIR}/PKGBUILD)

# Set a special flag if the environment is windows (should do the same in a config.g file)
IF (WIN32)
  ADD_DEFINITIONS(-D_WINDOWS=1)
ENDIF (WIN32)

######################################################################################

######################################################################################


######################################################################################
### compilation of examples?
######################################################################################

#SET(ENABLE_CMAKE_EXAMPLE TRUE CACHE BOOL "Enable copy of benchs and parameters file?")

######################################################################################


######################################################################################
### 4) Test config
######################################################################################

#IF (ENABLE_CMAKE_TESTING)
#  ENABLE_TESTING()
#ENDIF (ENABLE_CMAKE_TESTING)

######################################################################################


######################################################################################
### 5) Where must cmake go now ?
######################################################################################

#ADD_SUBDIRECTORY(app)
ADD_SUBDIRECTORY(doc)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(tutorial)

######################################################################################


######################################################################################
### 6) Install pkg-config config file for EO
######################################################################################

SET(PCPREFIX "/usr")
SET(PCFLAGS "-leoutils -leo -les -lga -lcma -lgcov")
SET(PCINCLUDEDIR "eo")
CONFIGURE_FILE(pc.cmake ${CMAKE_CURRENT_BINARY_DIR}/eo.pc)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/eo.pc DESTINATION local/${LIB}/pkgconfig COMPONENT headers)

######################################################################################


######################################################################################
### 7) Include packaging
######################################################################################

#INCLUDE(Packaging.cmake)

######################################################################################
