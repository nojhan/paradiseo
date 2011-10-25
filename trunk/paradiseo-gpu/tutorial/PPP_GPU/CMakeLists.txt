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

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib ${CUDA_LIB_DIR} )	

######################################################################################

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################	
			
CUDA_ADD_EXECUTABLE(GPUtestSimpleTS_PPP testSimpleTS_PPP.cu) 

TARGET_LINK_LIBRARIES(GPUtestSimpleTS_PPP eoutils ga eo)



