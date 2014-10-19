project(cmaes)


#find_package(Eigen3 REQUIRED)
#include_directories(EIGEN3_INCLUDE_DIR)
include_directories( ${EIGEN3_INCLUDE_DIR} )
#message( "MESSAGE:" ${EIGEN3_INCLUDE_DIR} )

#find_package(Boost 1.33.0)
#include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

#include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#include_directories(${EO_SRC_DIR}/src)
#include_directories(${EDO_SRC_DIR}/src)
#link_directories(${EO_BIN_DIRS}/${LIB})
#link_directories(${EDO_BIN_DIRS}/${LIB})

#   set(RESOURCES
#     ${PROJECT_NAME}.param
#     )
#
#   foreach(file ${RESOURCES})
#     execute_process(
#       COMMAND ${CMAKE_COMMAND} -E copy_if_different
#       ${CMAKE_CURRENT_SOURCE_DIR}/${file}
#       ${EDO_BIN_DIR}/${file}
#       )
#   endforeach(file)

#file(GLOB SOURCES *.cpp)

set(EXECUTABLE_OUTPUT_PATH ${EDO_BIN_DIR})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} eo eoutils edoutils ${Boost_LIBRARIES})

