######################################################################################
### 1) Where do we go now ?!?
######################################################################################

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  )

ADD_SUBDIRECTORY(common) # Rosenbrock and Sphere
#ADD_SUBDIRECTORY(eda_sa)
ADD_SUBDIRECTORY(eda)
ADD_SUBDIRECTORY(cmaes)

######################################################################################
