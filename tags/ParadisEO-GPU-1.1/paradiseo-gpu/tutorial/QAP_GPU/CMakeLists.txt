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
			
CUDA_ADD_EXECUTABLE(GPU_testSimpleHC testSimpleHC.cu) 
CUDA_ADD_EXECUTABLE(GPU_testSimpleTS testSimpleTS.cu) 

######################################################################################
### 4) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(GPU_testSimpleHC eoutils ga eo)
TARGET_LINK_LIBRARIES(GPU_testSimpleTS eoutils ga eo)
