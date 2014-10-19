# Lesson 1

######################################################################################
### 0) Define files
######################################################################################

set(files
    lesson1_eoEasyEA
    )

######################################################################################
### 1) Create the lesson 
######################################################################################
    
add_lesson(smp Lesson1 "${files}")

######################################################################################
### 2) Include dependencies
######################################################################################

#include_directories(${EO_SRC_DIR}/src
#					${SMP_SRC_DIR}/src
#					${PROBLEMS_SRC_DIR})


execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/lesson1_data.dat
    ${CMAKE_CURRENT_BINARY_DIR}/lesson1_data.dat)
