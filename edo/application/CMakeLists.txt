######################################################################################
### 1) Where do we go now ?!?
######################################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  )

include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(common) # Rosenbrock and Sphere
#add_subdirectory(eda_sa)

if(EIGEN3_FOUND) # see edoNormalAdaptive
    add_subdirectory(eda)
    add_subdirectory(cmaes)
endif()

######################################################################################
