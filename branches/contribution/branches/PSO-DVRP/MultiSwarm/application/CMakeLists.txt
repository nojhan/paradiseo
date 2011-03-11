



ADD_CUSTOM_TARGET(install DEPENDS ${PEOPSODVRP_SOURCE_DIR}/application/param ${PEOPSODVRP_SOURCE_DIR}/application/schema.xml)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${PEOPSODVRP_SOURCE_DIR}/application/param
      ${PEOPSODVRP_BINARY_DIR}/application)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${PEOPSODVRP_SOURCE_DIR}/application/schema.xml
      ${PEOPSODVRP_BINARY_DIR}/application)


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

link_directories (${PEOPSODVRP_BINARY_DIR}/src)
# Make sure the linker can find the Hello library once it is built.

ADD_EXECUTABLE(psoParallelMainDVRP  psoMain.cpp)

ADD_DEPENDENCIES(psoParallelMainDVRP  peo rmc_mpi)

######################################################################################


######################################################################################
### 5) Link the librairies for your executable
######################################################################################

# Only if you need to link libraries

TARGET_LINK_LIBRARIES(psoParallelMainDVRP   event )

# ${XML2_LIBS} peo rmc_mpi peo)

TARGET_LINK_LIBRARIES(psoParallelMainDVRP ${XML2_LIBS} peo rmc_mpi eo eoutils peo)


