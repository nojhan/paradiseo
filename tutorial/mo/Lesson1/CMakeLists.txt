# Lesson 1

######################################################################################
### 0) Define files
######################################################################################

set(files
    lesson1_simpleHC 
    lesson1_firstImprHC 
    lesson1_randomBestHC 
    lesson1_neutralHC 
    lesson1_iterContinuator 
    lesson1_fitContinuator 
    lesson1_fullEvalContinuator 
    lesson1_evalContinuator 
    lesson1_combinedContinuator 
    firstImprHC_maxSAT 
    )

######################################################################################
### 1) Create the lesson 
######################################################################################
    
add_lesson(mo Lesson1 "${files}")

######################################################################################
### 2) Include dependencies
######################################################################################

include_directories(
    #${EO_SRC_DIR}/src
    #${MO_SRC_DIR}/src
	#${PROBLEMS_SRC_DIR}
)

