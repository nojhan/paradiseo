######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${CMAKE_INSTALL_PREFIX}/include/paradiseo/eo)
INCLUDE_DIRECTORIES(${CMAKE_INSTALL_PREFIX}/include/paradiseo/mo)
INCLUDE_DIRECTORIES(${CMAKE_INSTALL_PREFIX}/include/paradiseo/moeo)
INCLUDE_DIRECTORIES(${CMAKE_INSTALL_PREFIX}/include/paradiseo/peo)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(/usr/local/include/GTL)
LINK_DIRECTORIES(${CMAKE_INSTALL_PREFIX}/lib)
LINK_DIRECTORIES(/usr/local/lib)

######################################################################################
### 2) Define PhyloMOEA  target
######################################################################################


SET( PHYLOMOEA_SOURCES	eigensolver.cpp
						likelihoodcalculator.cpp
						likoptimizer.cpp
						matrixutils.cpp
						parsimonycalculator.cpp
						PhyloMOEA.cpp
						probmatrixcontainer.cpp
						ProbMatrix.cpp
						RandomNr.cpp
						Sequences.cpp
						SubsModel.cpp
						phylotreeIND.cpp
						treeIterator.cpp
						utils.cpp 
						PhyloMOEO_packunpack.cpp)

SET( TESTOMP_SOURCES	eigensolver.cpp
						likelihoodcalculator.cpp
						matrixutils.cpp
						probmatrixcontainer.cpp
						ProbMatrix.cpp
						RandomNr.cpp
						Sequences.cpp
						SubsModel.cpp
						phylotreeIND.cpp
						treeIterator.cpp
						utils.cpp 
						likoptimizer.cpp
						testomp.cpp
)


ADD_EXECUTABLE( PhyloMOEA ${PHYLOMOEA_SOURCES} )
ADD_EXECUTABLE( testomp   ${TESTOMP_SOURCES} )

TARGET_LINK_LIBRARIES(PhyloMOEA gsl gslcblas GTL eo eoutils ga moeo cma peo rmc_mpi xml2)
TARGET_LINK_LIBRARIES(testomp gsl gslcblas GTL eo eoutils ga moeo cma peo rmc_mpi xml2 gomp)

INSTALL( TARGETS PhyloMOEA RUNTIME DESTINATION bin)