

#####################################################################################
### Include required modules / configuration files
#####################################################################################

# For eo::mpi
ENABLE_LANGUAGE(C)

# For openmp parallel
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

INCLUDE(FindGnuplot)

# Set a special flag if the environment is windows (should do the same in a config.g file)
IF (WIN32)
  ADD_DEFINITIONS(-D_WINDOWS=1)
ENDIF (WIN32)



#####################################################################################
### Include required modules / configuration files
#####################################################################################

# now create config headers
CONFIGURE_FILE(config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)

# now create config install_symlink script file
CONFIGURE_FILE(install_symlink.py.cmake ${CMAKE_CURRENT_BINARY_DIR}/install_symlink.py)


######################################################################################
### Include subdirectories
######################################################################################


add_subdirectory(doc)
add_subdirectory(src)

if(ENABLE_CMAKE_TESTING)
    add_subdirectory(test)
endif(ENABLE_CMAKE_TESTING)

if(ENABLE_CMAKE_EXAMPLE)
    if(${CMAKE_VERBOSE_MAKEFILE})
        message("EO examples:")
    endif(${CMAKE_VERBOSE_MAKEFILE})
    add_subdirectory(tutorial)
    add_subdirectory(app)
endif(ENABLE_CMAKE_EXAMPLE)


