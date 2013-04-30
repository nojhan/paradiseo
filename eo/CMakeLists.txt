

#####################################################################################
### Include required modules / configuration files
#####################################################################################

# For eo::mpi
enable_language(C)

# For openmp parallel
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
      message( "ERROR: You asked for OpenMP but it has not been found." )
      set(IS_FATAL 1)
  endif(OPENMP_FOUND)
endif(ENABLE_OPENMP)

if(ENABLE_GNUPLOT)
    include(FindGnuplot)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_GNUPLOT -DGNUPLOT_PROGRAM=\\\"${GNUPLOT_EXECUTABLE}\\\"")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_GNUPLOT -DGNUPLOT_PROGRAM=\\\"${GNUPLOT_EXECUTABLE}\\\"")
endif(ENABLE_GNUPLOT)

# set a special flag if the environment is windows (should do the same in a config.g file)
if (WIN32)
  add_definitions(-D_WINDOWS=1)
endif (WIN32)



#####################################################################################
### Include required modules / configuration files
#####################################################################################

# now create config headers
configure_file(config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)

# now create config install_symlink script file
configure_file(install_symlink.py.cmake ${CMAKE_CURRENT_BINARY_DIR}/install_symlink.py)


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
    #add_subdirectory(app) # FIXME must be updated
endif(ENABLE_CMAKE_EXAMPLE)

