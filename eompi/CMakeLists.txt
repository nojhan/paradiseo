######################################################################################
### Include subdirectories
######################################################################################
#add_subdirectory(doc)
add_subdirectory(src)

if(ENABLE_CMAKE_TESTING)
    add_subdirectory(test)
endif(ENABLE_CMAKE_TESTING)