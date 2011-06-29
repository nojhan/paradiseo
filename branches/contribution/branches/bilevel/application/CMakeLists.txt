######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src
                    ${PARADISEO_MOEO_SRC_DIR}/src
                    ${PARADISEO_MO_SRC_DIR}/src
                    ${CMAKE_SOURCE_DIR}/tsp/
                    ${CMAKE_SOURCE_DIR}/src)

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib)
LINK_DIRECTORIES(${PARADISEO_MOEO_BIN_DIR}/lib)
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/lib)
######################################################################################


######################################################################################
### 3) Define your target: just an executable here
######################################################################################





ADD_EXECUTABLE(mbiMDVRP mbiMDVRP.cpp)
ADD_EXECUTABLE(simpletemp simpletemp.cpp)
ADD_EXECUTABLE(simpleRepair simpleRepair.cpp)
ADD_EXECUTABLE(multitemp multitemp.cpp)
ADD_EXECUTABLE(multirepair multirepair.cpp)
ADD_EXECUTABLE(bilevelGen bilevelGen.cpp)
ADD_EXECUTABLE(verifRat verifRat.cpp)
ADD_EXECUTABLE(verifRat2 verifRat2.cpp)
ADD_EXECUTABLE(verifRatmulti verifRatmulti.cpp)
ADD_EXECUTABLE(verifRatmulti2 verifRatmulti2.cpp)
######################################################################################


######################################################################################
### 4) Link the librairies for your executable
######################################################################################

# Only if you need to link libraries
TARGET_LINK_LIBRARIES(mbiMDVRP eo eoutils lvrp lbeo)
TARGET_LINK_LIBRARIES(simpletemp eo eoutils lbeo lvrp )
TARGET_LINK_LIBRARIES(simpleRepair eo eoutils lbeo lvrp )
TARGET_LINK_LIBRARIES(multitemp eo eoutils lbeo lvrp )
TARGET_LINK_LIBRARIES(multirepair eo eoutils lbeo lvrp )
TARGET_LINK_LIBRARIES(bilevelGen eo eoutils lvrp lbeo)
TARGET_LINK_LIBRARIES(verifRat eo eoutils lvrp lbeo)
TARGET_LINK_LIBRARIES(verifRat2 eo eoutils lvrp lbeo)
TARGET_LINK_LIBRARIES(verifRatmulti eo eoutils lvrp lbeo)
TARGET_LINK_LIBRARIES(verifRatmulti2 eo eoutils lvrp lbeo)

######################################################################################


######################################################################################
### 5) Copy the instances and the "param" file in the build path for an easy use.
### 
### --> run the "make install" target to copy the parameter file / instances
###	in the directory where you build the application
######################################################################################

#ADD_CUSTOM_TARGET(install DEPENDS ${CMAKE_SOURCE_DIR}/application/NSGAII.param
#				  ${CMAKE_SOURCE_DIR}/application)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${CMAKE_SOURCE_DIR}/application/NSGAII.param
      ${CMAKE_BINARY_DIR}/application)

######################################################################################
