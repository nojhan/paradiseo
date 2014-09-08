######################################################################################
### 0) Include directories
######################################################################################

#include_directories(${EO_SRC_DIR})
#include_directories(${MO_SRC_DIR}/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 1) Set paths
######################################################################################

set(MOEO_LIB_OUTPUT_PATH ${MOEO_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${MOEO_LIB_OUTPUT_PATH})

######################################################################################
### 2) Build lib
######################################################################################

set (MOEO_CORE core/moeoObjectiveVectorTraits.cpp)

add_library(moeo STATIC ${MOEO_CORE})

set(MOEO_VERSION ${GLOBAL_VERSION})
set_target_properties(moeo PROPERTIES VERSION "${MOEO_VERSION}")

install(TARGETS moeo ARCHIVE DESTINATION ${LIB} COMPONENT libraries)

######################################################################################
### 3) Look for headers
######################################################################################

install(FILES moeo.h DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG} COMPONENT headers)

######################################################################################
### 4) Install directories
######################################################################################

install(DIRECTORY algo archive comparator continue core distance diversity do explorer fitness hybridization metric replacement scalarStuffs selection utils
        DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/moeo
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h"
        )

