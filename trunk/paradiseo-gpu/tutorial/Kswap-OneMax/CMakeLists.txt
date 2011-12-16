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
			
CUDA_ADD_EXECUTABLE(GPUtestKswapHCByModif testKswapHCByModif.cu)
CUDA_ADD_EXECUTABLE(GPUtestKswapHCByCpy testKswapHCByCpy.cu)  

TARGET_LINK_LIBRARIES(GPUtestKswapHCByModif eoutils ga eo)
TARGET_LINK_LIBRARIES(GPUtestKswapHCByCpy eoutils ga eo)

######################################################################################
