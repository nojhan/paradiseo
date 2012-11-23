project(eda)

find_package(Boost 1.33.0)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

include_directories(${EO_SRC_DIR}/src)
include_directories(${EDO_SRC_DIR}/src)
link_directories(${EO_BIN_DIR}/${LIB})
link_directories(${EDO_BIN_DIR}/${LIB})

set(RESOURCES
  ${PROJECT_NAME}.param
  )

foreach(FILE ${RESOURCES})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    ${EDO_BIN_DIR}/${FILE}
    )
endforeach(FILE)

file(GLOB SOURCES *.cpp)

set(EXECUTABLE_OUTPUT_PATH ${EDO_BIN_DIR})

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} eo eoutils edoutils ${Boost_LIBRARIES})

