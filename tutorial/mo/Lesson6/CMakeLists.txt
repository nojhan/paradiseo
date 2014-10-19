# Lesson 6

######################################################################################
### 0) Define files
######################################################################################

set(files
    adaptiveWalks
    autocorrelation
    densityOfStates
    fdc
    fitnessCloud
    neutralDegree
    neutralWalk
    sampling
    testMetropolisHasting
    testRandomNeutralWalk
    testRandomWalk
    )
    
######################################################################################
### 1) Create the lesson 
######################################################################################

add_lesson(mo Lesson6 "${files}")

######################################################################################
### 2) Include dependencies
######################################################################################

include_directories(
    #${EO_SRC_DIR}/src
	#${MO_SRC_DIR}/src
	#${PROBLEMS_SRC_DIR}
)
