######################################################################################
### 0) Include lessons subdirectories
######################################################################################

#include_directories(${EO_SRC_DIR}/src)
#include_directories(${MOEO_SRC_DIR}/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define output paths
######################################################################################

set(FLOWSHOP_LIB_OUTPUT_PATH ${FLOWSHOP_BINARY_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${FLOWSHOP_LIB_OUTPUT_PATH})

######################################################################################
### 3) Set flowshop sources
######################################################################################

set(FLOWSHOP_SOURCES   
        FlowShopBenchmarkParser.cpp 
        FlowShopEval.cpp 
        FlowShopObjectiveVectorTraits.cpp 
        FlowShopOpCrossoverQuad.cpp 
        FlowShop.cpp)

######################################################################################
### 4) Create flowshop library
######################################################################################

add_library(flowshop STATIC ${FLOWSHOP_SOURCES})
add_dependencies(flowshop moeo)

target_link_libraries(flowshop eo)

install(TARGETS flowshop ARCHIVE DESTINATION ${LIB} COMPONENT libraries)

######################################################################################
### 5) Installing benchs
######################################################################################

install(DIRECTORY benchs DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/moeo/tutorial/flowshop COMPONENT tutorial)

######################################################################################
### 6) Look for headers
######################################################################################

file(GLOB HDRS *.h)
install(FILES ${HDRS} DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/moeo/tutorial/examples/flowshop COMPONENT headers)
