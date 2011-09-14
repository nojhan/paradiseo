#########################################################################################################
# 1) ParadisEO-GPU install: SIMPLE Configuration
#########################################################################################################

#  Here, just specify PARADISEO_DIR : the directory where ParadisEO has been installed
SET(PARADISEO_DIR "Set ParadisEO directory" CACHE PATH "ParadisEO directory" FORCE)

#  Here, just specify CUDA_DIR : the directory where CUDA has been installed
SET(CUDA_DIR "Set CUDA directory" CACHE PATH "CUDA directory" FORCE)

#  Here, just specify NVIDIA_DIR : the directory where NVIDIA header has been installed
SET(NVIDIA_DIR "Set NVIDIA directory" CACHE PATH "NVIDIA directory" FORCE)

#  Here, just specify GPU_DIR : the directory where package Paradiseo GPU has been extrated
SET(PARADISEO_GPU_DIR "Set ParadisEO-GPU directory" CACHE PATH "ParadisEO-GPU directory" FORCE)

# Here, just specify the name of cutil library depending to the used NVIDIA_GPU_Computing_SDK
# Find in NVIDIA_GPU_Computing_SDK/C/lib
SET(cutil "Set cutil library name")

#########################################################################################################
# 2) ParadisEO-GPU install: ADVANCED Configuration
#########################################################################################################

SET(NVIDIA_SRC_DIR "${NVIDIA_DIR}/C/common/inc" CACHE PATH "NVIDIA source directory" FORCE)
SET(NVIDIA_LIB_DIR "${NVIDIA_DIR}/C/lib" CACHE PATH "NVIDIA library directory" FORCE)

SET(CUDA_SRC_DIR "${CUDA_DIR}/include" CACHE PATH "CUDA source directory" FORCE)
SET(CUDA_LIB_DIR "${CUDA_DIR}/lib" CACHE PATH "CUDA library directory" FORCE)

SET(PARADISEO_EO_SRC_DIR "${PARADISEO_DIR}/paradiseo-eo" CACHE PATH "ParadisEO-EO source directory" FORCE)
SET(PARADISEO_EO_BIN_DIR "${PARADISEO_DIR}/paradiseo-eo/build" CACHE PATH "ParadisEO-EO binary directory" FORCE)

SET(PARADISEO_MO_SRC_DIR "${PARADISEO_DIR}/paradiseo-mo" CACHE PATH "ParadisEO-MO source directory" FORCE)
SET(PARADISEO_MO_BIN_DIR "${PARADISEO_DIR}/paradiseo-mo/build" CACHE PATH "ParadisEO-MO binary directory" FORCE)

SET(PARADISEO_MOEO_SRC_DIR "${PARADISEO_DIR}/paradiseo-moeo" CACHE PATH "ParadisEO-MOEO source directory" FORCE)
SET(PARADISEO_MOEO_BIN_DIR "${PARADISEO_DIR}/paradiseo-moeo/build" CACHE PATH "ParadisEO-MOEO binary directory" FORCE)

SET(PARADISEO_PEO_SRC_DIR "${PARADISEO_DIR}/paradiseo-peo" CACHE PATH "ParadisEO-PEO source directory" FORCE)
SET(PARADISEO_PEO_BIN_DIR "${PARADISEO_DIR}/paradiseo-peo/build" CACHE PATH "ParadisEO-PEO binary directory" FORCE)

SET(PARADISEO_GPU_SRC_DIR "${PARADISEO_GPU_DIR}/src" CACHE PATH "ParadisEO-GPU source directory" FORCE)
SET(PARADISEO_GPU_BIN_DIR "${PARADISEO_GPU_DIR}/build" CACHE PATH "ParadisEO-GPU binary directory" FORCE)

SET(PARADISEO_PROBLEMS_SRC_DIR "${PARADISEO_DIR}/problems" CACHE PATH "ParadisEO-problems source directory" FORCE)

#########################################################################################################
