
INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src
                   ${PARADISEO_MO_SRC_DIR}/src
                   ${PARADISEO_MOEO_SRC_DIR}/src
                   ${PARADISEO_PEO_SRC_DIR}/src
		            ${PEOPSODVRP_SOURCE_DIR}/src)


######################################################################################


######################################################################################
###  Specify where CMake can find the libraries
######################################################################################

# Link with the EO library directory, only if you need it
LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib)

# Link with the MOEO library directory, only if you need it
LINK_DIRECTORIES(${PARADISEO_MOEO_BIN_DIR}/lib)

# Link with the PEO library directory, only if you need it
LINK_DIRECTORIES(${PARADISEO_PEO_BIN_DIR}/lib)

LINK_DIRECTORIES(${PEOPSODVRP_BINARY_DIR}/src)


# Make sure the linker can find the Hello library once it is built.

add_library (event eoEventScheduler.cpp parser.cpp eoPsoDVRPutils.cpp eoPsoDVRP.cpp eoGlobal.cpp)


