######################################################################################
### 0) Where must cmake go now ?
######################################################################################

#add_subdirectory(Lesson1)
add_subdirectory(Lesson2)
#add_subdirectory(Lesson3)
#add_subdirectory(Lesson4)
add_subdirectory(Lesson5)
#add_subdirectory(Lesson6)

######################################################################################
### 1) Install and copy benchs
######################################################################################
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/benchs
            ${CMAKE_CURRENT_BINARY_DIR}/benchs)
            
install(DIRECTORY benchs DESTINATION include${INSTALL_SUB_DIR}/peo/tutorial COMPONENT tutorial)
