######################################################################################
### 1) Set all needed source files for the project
######################################################################################

FILE(GLOB SOURCES *.cpp)

SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
ADD_LIBRARY(edoutils ${SOURCES})
INSTALL(TARGETS edoutils ARCHIVE DESTINATION lib COMPONENT libraries)

FILE(GLOB HDRS *.h utils)
INSTALL(FILES ${HDRS} DESTINATION include/edo/utils COMPONENT headers)

######################################################################################
