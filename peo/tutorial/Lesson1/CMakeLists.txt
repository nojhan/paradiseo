######################################################################################
### 0) Set the compiler and define targets to easily run the lessons
######################################################################################

set(CMAKE_CXX_COMPILER mpicxx)

#ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson1/param ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson1/schema.xml)

execute_process(
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${CMAKE_CURRENT_SOURCE_DIR}/param
      	${CMAKE_CURRENT_BINARY_DIR}
  	COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/schema.xml
        ${CMAKE_CURRENT_BINARY_DIR}
)

######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src 
                    ${MOEO_SRC_DIR}/src  
                    ${MO_SRC_DIR}/src 
                    ${PEO_SRC_DIR}/src
                    )

######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

add_executable(multistartMO main.cpp)

######################################################################################
### 5) Link the librairies
######################################################################################

target_link_libraries(multistartMO xml2 peo rmc_mpi eo eoutils peo) 

