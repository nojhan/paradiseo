######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(

  # include EO source directory
  ${EO_SRC_DIR}/src
  # include MO source directory
  ${MO_SRC_DIR}/src
  # include problems  directory
  ${PROBLEMS_SRC_DIR}
  # include GPU  directory
  ${GPU_SRC_DIR}/src
  # include your source directory
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  )

######################################################################################

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${CUDA_DIR}/lib )	

######################################################################################

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################	

CUDA_ADD_EXECUTABLE(GPUtestFirstImpr testFirstImpr.cu) 
CUDA_ADD_EXECUTABLE(GPUtestNeutralHC testNeutralHC.cu) 
CUDA_ADD_EXECUTABLE(GPUtestSimpleHC testSimpleHC.cu) 
CUDA_ADD_EXECUTABLE(GPUtestSimpleHCByCpy testSimpleHCByCpy.cu) 
CUDA_ADD_EXECUTABLE(GPUtestSimpleTS testSimpleTS.cu) 
CUDA_ADD_EXECUTABLE(GPUtestSimulatedAnnealing testSimulatedAnnealing.cu) 

TARGET_LINK_LIBRARIES(GPUtestFirstImpr eoutils ga eo)
TARGET_LINK_LIBRARIES(GPUtestNeutralHC eoutils ga eo)
TARGET_LINK_LIBRARIES(GPUtestSimpleHC eoutils ga eo)
TARGET_LINK_LIBRARIES(GPUtestSimpleHCByCpy eoutils ga eo)
TARGET_LINK_LIBRARIES(GPUtestSimpleTS eoutils ga eo)
TARGET_LINK_LIBRARIES(GPUtestSimulatedAnnealing eoutils ga eo)

######################################################################################
