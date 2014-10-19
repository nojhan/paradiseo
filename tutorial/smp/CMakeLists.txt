######################################################################################
### 0) Include lessons subdirectories
######################################################################################

add_subdirectory(Lesson1)
add_subdirectory(Lesson2)
add_subdirectory(Lesson3)
add_subdirectory(Lesson4)

######################################################################################
### 0) Install BaseFile for all lessons and data
######################################################################################

install(FILES BaseLesson.h DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/smp/tutorial COMPONENT tutorial)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/lessonData.dat
    ${CMAKE_CURRENT_BINARY_DIR}/lessonData.dat)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lessonData.dat DESTINATION share/${PROJECT_TAG}/smp/tutorial COMPONENT tutorial)
