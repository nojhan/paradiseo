FIND_PACKAGE(Boost 1.33.0 REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(RESOURCES
  cma_sa.param
  plot.py
  )

FOREACH(file ${RESOURCES})
  EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/${file}
    ${CMAKE_CURRENT_BINARY_DIR}/${file}
    )
ENDFOREACH(file)

ADD_EXECUTABLE(cma_sa main.cpp)

TARGET_LINK_LIBRARIES(cma_sa
  ${Boost_LIBRARIES}
  BOPO
  eoutils
  pthread
  moeo
  eo
  peo
  rmc_mpi
  eometah
  nklandscapes
  BOPO
  #${MPICH2_LIBRARIES}
  ${LIBXML2_LIBRARIES}
  ${MPI_LIBRARIES}
  )
