######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(

  # include CUDA source directory
  ${CUDA_SRC_DIR}
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

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib ${CUDA_LIB_DIR} )	

######################################################################################

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################	
			
CUDA_ADD_EXECUTABLE(CutestSimpleTS_CPU testSimpleTS_CPU.cu) 			
CUDA_ADD_EXECUTABLE(CutestSimpleHC_CPU testSimpleHC_CPU.cu)

TARGET_LINK_LIBRARIES(CutestSimpleTS_CPU eoutils ga eo)
TARGET_LINK_LIBRARIES(CutestSimpleHC_CPU eoutils ga eo)