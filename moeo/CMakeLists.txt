######################################################################################
### 0) Include subdirectories
######################################################################################

add_subdirectory(doc)
add_subdirectory(src)

if(ENABLE_CMAKE_TESTING)
    add_subdirectory(test)
endif(ENABLE_CMAKE_TESTING)

if(ENABLE_CMAKE_EXAMPLE)
    if(${CMAKE_VERBOSE_MAKEFILE})
        message("MOEO Examples :")
    endif(${CMAKE_VERBOSE_MAKEFILE})
    add_subdirectory(tutorial)
endif(ENABLE_CMAKE_EXAMPLE)
