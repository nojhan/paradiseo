######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(

  # include CUDA source directory
  ${GPU_SRC_DIR}
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
			
CUDA_ADD_EXECUTABLE(testSimpleTS_CPU testSimpleTS_CPU.cu) 			
CUDA_ADD_EXECUTABLE(testSimpleHC_CPU testSimpleHC_CPU.cu)

TARGET_LINK_LIBRARIES(testSimpleTS_CPU eoutils ga eo)
TARGET_LINK_LIBRARIES(testSimpleHC_CPU eoutils ga eo)