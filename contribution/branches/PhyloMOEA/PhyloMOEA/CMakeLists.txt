######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${CMAKE_INSTALL_PREFIX}/include/paradiseo/eo)
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
						utils.cpp )

ADD_EXECUTABLE( PhyloMOEA ${PHYLOMOEA_SOURCES} )

TARGET_LINK_LIBRARIES(PhyloMOEA gsl gslcblas GTL eo eoutils ga moeo cma)