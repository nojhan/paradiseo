#######################################################################################
### Paths to EO, MO, GPU, MOEO and CUDA must be specified above.
#######################################################################################

SET(EO_SRC_DIR "${CMAKE_SOURCE_DIR}/../paradiseo-eo" CACHE PATH "ParadisEO-EO source directory" FORCE)
SET(EO_BIN_DIR "${CMAKE_BINARY_DIR}/../../paradiseo-eo/build" CACHE PATH "ParadisEO-EO binary directory" FORCE)
    
SET(MO_SRC_DIR "${CMAKE_SOURCE_DIR}/../paradiseo-mo" CACHE PATH "ParadisMO-MO source directory" FORCE)
SET(MO_BIN_DIR "${CMAKE_BINARY_DIR}/../../paradiseo-mo/build" CACHE PATH "ParadisMO-MO binary directory" FORCE)

SET(GPU_SRC_DIR "${CMAKE_SOURCE_DIR}/../paradiseo-gpu" CACHE PATH "ParadisEO-GPU source directory" FORCE)
SET(GPU_BIN_DIR "${CMAKE_BINARY_DIR}/../../paradiseo-gpu/build" CACHE PATH "ParadisEO-GPU binary directory" FORCE)


SET(OLDMO_SRC_DIR "${CMAKE_SOURCE_DIR}/../paradiseo-old-mo" CACHE PATH "ParadisMO-MO source directory" FORCE)
SET(OLDMO_BIN_DIR "${CMAKE_BINARY_DIR}/../../paradiseo-old-mo/build" CACHE PATH "ParadisMO-MO binary directory" FORCE)

SET(MOEO_SRC_DIR "${CMAKE_SOURCE_DIR}/../paradiseo-moeo" CACHE PATH "ParadisMOEO-MOEO source directory" FORCE)
SET(MOEO_BIN_DIR "${CMAKE_BINARY_DIR}/../../paradiseo-moeo/build" CACHE PATH "ParadisMOEO-MOEO binary directory" FORCE)

SET(PROBLEMS_SRC_DIR "${CMAKE_SOURCE_DIR}/../problems" CACHE PATH "Problems dependant source directory" FORCE)
    
SET(CUDA_DIR "/usr/local/cuda" CACHE PATH "cuda directory" FORCE)

    
