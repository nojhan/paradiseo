######################################################################################
### 1) Set all needed source files for the project
######################################################################################

include_directories(${EO_SRC_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(EDO_LIB_OUTPUT_PATH ${EDO_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${EDO_LIB_OUTPUT_PATH})


######################################################################################
### 3) Look for headers
######################################################################################

file(GLOB HDRS edo)
install(FILES ${HDRS} DESTINATION include${INSTALL_SUB_DIR}/edo COMPONENT headers)


######################################################################################
### 4) Install directories
######################################################################################

install(DIRECTORY utils
        DESTINATION include${INSTALL_SUB_DIR}/edo
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h"
        )


######################################################################################
### 2) Where must cmake go now ?
######################################################################################

ADD_SUBDIRECTORY(utils)

######################################################################################
