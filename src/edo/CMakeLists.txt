######################################################################################
### 1) Set all needed source files for the project
######################################################################################

#include_directories(${EO_SRC_DIR})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(EDO_LIB_OUTPUT_PATH ${EDO_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${EDO_LIB_OUTPUT_PATH})


######################################################################################
### 3) Look for headers
######################################################################################

#file(GLOB HDRS edo.h)
# Indication from CMake : "We do not recommend using GLOB to collect a list of 
#                         sources files from your source tree. If no 
#                         CMakeLists.txt file changes when a source is added or 
#                         removed then the generated build system cannot know 
#                         when to ask CMake to generate."

set(HDRS
	edoAlgoAdaptive.h
	edoAlgo.h
	edoAlgoStateless.h
	edoBinomial.h
	edoBinomialMulti.h
	edoBounderBound.h
	edoBounder.h
	edoBounderNo.h
	edoBounderRng.h
	edoBounderUniform.h
	edoContinue.h
	edoDistrib.h
	edoEDASA.h
	edoEstimatorAdaptive.h
	edoEstimatorBinomial.h
	edoEstimatorBinomialMulti.h
	edoEstimator.h
	edoEstimatorNormalAdaptive.h
	edoEstimatorNormalMono.h
	edoEstimatorNormalMulti.h
	edoEstimatorUniform.h
	#edo.h
	edoModifierDispersion.h
	edoModifier.h
	edoModifierMass.h
	edoNormalAdaptive.h
	edoNormalMonoCenter.h
	edoNormalMono.h
	edoNormalMultiCenter.h
	edoNormalMulti.h
	edoRepairerApply.h
	edoRepairerDispatcher.h
	edoRepairer.h
	edoRepairerModulo.h
	edoRepairerRound.h
	edoSamplerBinomial.h
	edoSamplerBinomialMulti.h
	edoSampler.h
	edoSamplerNormalAdaptive.h
	edoSamplerNormalMono.h
	edoSamplerNormalMulti.h
	edoSamplerUniform.h
	edoTransform.h
	edoUniformCenter.h
	edoUniform.h
	edoVectorBounds.h
)

install(FILES edo.h DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG} COMPONENT headers)

install(FILES ${HDRS} DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/edo COMPONENT headers)

######################################################################################
### 4) Install directories
######################################################################################

install(DIRECTORY utils
        DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/edo
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h"
        )


######################################################################################
### 2) Where must cmake go now ?
######################################################################################

ADD_SUBDIRECTORY(utils)

######################################################################################
