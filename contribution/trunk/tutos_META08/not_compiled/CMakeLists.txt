######################################################################################
### 1) Project properties
######################################################################################

# set the project name
PROJECT(TUTORIAUX)

# check cmake version compatibility
CMAKE_MINIMUM_REQUIRED(VERSION 2.4 FATAL_ERROR)

# regular expression checking
INCLUDE_REGULAR_EXPRESSION("^.*$" "^$")

# set a language
ENABLE_LANGUAGE(CXX)
ENABLE_LANGUAGE(C)

#####################################################################################


######################################################################################
### 2) Include the main configuration variables
######################################################################################

# Need the config file whose full path is given thanks to the "config" variable
INCLUDE("install.cmake")

######################################################################################


######################################################################################
### 3) Paths checking 
######################################################################################

IF(WIN32)
  SET (ABSOLUTE_PATH_REGEX "^[A-Z]:|^[a-z]:")
ELSE(WIN32)
  SET (ABSOLUTE_PATH_REGEX "^/")
ENDIF(WIN32)

SET(REQUIRED_PATHS "PARADISEO_EO_SRC_DIR" "PARADISEO_EO_BIN_DIR" "PARADISEO_MO_SRC_DIR" "PARADISEO_MO_BIN_DIR" "PARADISEO_MO_SRC_DIR" "PARADISEO_MOEO_BIN_DIR")
FOREACH (path ${REQUIRED_PATHS})
	  	IF(EXISTS ${${path}})
			MESSAGE (STATUS "Using ${path}=${${path}}")
		ELSE(EXISTS ${${path}})
			MESSAGE (FATAL_ERROR "\n Cannot find \"${${path}}\". Please, fill \"${config}\" with a correct value")
		ENDIF(EXISTS ${${path}})
		
		IF(NOT ${${path}} MATCHES "${ABSOLUTE_PATH_REGEX}")
			MESSAGE (FATAL_ERROR "${${path}} MUST BE an absolute path")
		ENDIF(NOT ${${path}} MATCHES "${ABSOLUTE_PATH_REGEX}")
ENDFOREACH (path ${REQUIRED_PATHS})
######################################################################################

ADD_CUSTOM_TARGET(installall DEPENDS
	${SOURCES_DIR}/tsp/benchs
	${SOURCES_DIR}/flowshop/benchs
	${SOURCES_DIR}/EO_lesson1/param
	${SOURCES_DIR}/MO_lesson1/param
	${SOURCES_DIR}/MO_lesson2/param
	${SOURCES_DIR}/MO_lesson3/param
	${SOURCES_DIR}/MO_lesson4/param
	${SOURCES_DIR}/MOEO_lesson/param
	${SOURCES_DIR}/hybrid_lesson/param	
)

ADD_CUSTOM_COMMAND(
    TARGET installall
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory 
      ${SOURCES_DIR}/tsp/benchs
      ${INSTALL_DIR}/tsp/benchs)

ADD_CUSTOM_COMMAND(
    TARGET installall
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory 
      ${SOURCES_DIR}/flowshop/benchs
      ${INSTALL_DIR}/flowshop/benchs)

ADD_CUSTOM_COMMAND(
    TARGET installall
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${SOURCES_DIR}/EO_lesson1/param
      ${INSTALL_DIR}/EO_lesson1/Release)

ADD_CUSTOM_COMMAND(
    TARGET installall
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${SOURCES_DIR}/MOEO_lesson/param
      ${INSTALL_DIR}/MOEO_lesson/Release)

ADD_CUSTOM_COMMAND(
    TARGET installall
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${SOURCES_DIR}/MO_lesson1/param
      ${INSTALL_DIR}/MO_lesson1/Release)

ADD_CUSTOM_COMMAND(
    TARGET installall
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${SOURCES_DIR}/MO_lesson2/param
      ${INSTALL_DIR}/MO_lesson2/Release)

ADD_CUSTOM_COMMAND(
    TARGET installall
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${SOURCES_DIR}/MO_lesson3/param
      ${INSTALL_DIR}/MO_lesson3/Release)

ADD_CUSTOM_COMMAND(
    TARGET installall
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${SOURCES_DIR}/MO_lesson4/param
      ${INSTALL_DIR}/MO_lesson4/Release)

ADD_CUSTOM_COMMAND(
    TARGET installall
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${SOURCES_DIR}/hybrid_lesson/param
      ${INSTALL_DIR}/hybrid_lesson/Release)


######################################################################################
### 6) Where must cmake go now ?
######################################################################################

SUBDIRS(tsp EO_lesson1 MO_lesson1 MO_lesson2 MO_lesson3 MO_lesson4 flowshop MOEO_lesson hybrid_lesson) 

######################################################################################

