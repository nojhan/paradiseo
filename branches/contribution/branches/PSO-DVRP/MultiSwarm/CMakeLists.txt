
######################################################################################
### 0) Include the install configuration file where are defined the main variables
######################################################################################

INCLUDE(${CMAKE_SOURCE_DIR}/install.cmake)


FIND_PROGRAM(XML2_CONFIG_PROGRAM NAMES xml2-config  PATHS /usr/local/bin /usr/bin  DOC "The libxml2 (www.xmlsoft.org) configuration tool")

IF(NOT XML2_CONFIG_PROGRAM)
	#MESSAGE(FATAL_ERROR "Libxml2 cannot be found on your system. Libxml2 is required for ParadisEO-PEO install")
ENDIF(NOT XML2_CONFIG_PROGRAM)

# set libxml2 cflags in XML2_CFLAGS
EXECUTE_PROCESS(COMMAND ${XML2_CONFIG_PROGRAM} --cflags  OUTPUT_VARIABLE XML2_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

# set libxml2 libs in XML2_LIBS
EXECUTE_PROCESS(COMMAND ${XML2_CONFIG_PROGRAM} --libs  OUTPUT_VARIABLE XML2_LIBS)
######################################################################################


######################################################################################
###  Set your application properties
######################################################################################

# Define your project name
PROJECT(PEOPSODVRP)

######################################################################################


######################################################################################
SUBDIRS ( src application )



######################################################################################

