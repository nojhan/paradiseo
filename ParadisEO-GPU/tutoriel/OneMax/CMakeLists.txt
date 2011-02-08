###############################################################################
##
## CMakeLists file for OneMax Example/application
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(

  # include CUDA source directory
  ${CUDA_SRC_DIR}
  # include NVIDIA source directory
  ${NVIDIA_SRC_DIR}
  # include EO source directory
  ${PARADISEO_EO_SRC_DIR}/src
  # include MO source directory
  ${PARADISEO_MO_SRC_DIR}/src
  # include problems  directory
  ${PARADISEO_PROBLEMS_SRC_DIR}
  # include GPU  directory
  ${PARADISEO_GPU_SRC_DIR}
  # include your source directory
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  )

######################################################################################

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib  ${NVIDIA_LIB_DIR} ${CUDA_LIB_DIR} )	

######################################################################################

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################	
			
CUDA_ADD_EXECUTABLE(CutestFirstImpr testFirstImpr.cu) 
CUDA_ADD_EXECUTABLE(CutestNeutralHC testNeutralHC.cu) 
CUDA_ADD_EXECUTABLE(CutestSimpleHC testSimpleHC.cu) 
CUDA_ADD_EXECUTABLE(CutestSimpleTS testSimpleTS.cu) 
CUDA_ADD_EXECUTABLE(CutestSimulatedAnnealing testSimulatedAnnealing.cu) 

TARGET_LINK_LIBRARIES(CutestFirstImpr eoutils ga eo ${cutil})
TARGET_LINK_LIBRARIES(CutestNeutralHC eoutils ga eo ${cutil})
TARGET_LINK_LIBRARIES(CutestSimpleHC eoutils ga eo ${cutil})
TARGET_LINK_LIBRARIES(CutestSimpleTS eoutils ga eo ${cutil})
TARGET_LINK_LIBRARIES(CutestSimulatedAnnealing eoutils ga eo ${cutil})

######################################################################################
