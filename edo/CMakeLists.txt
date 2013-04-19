
if(EDO_USE_LIB STREQUAL "uBLAS")
    find_package(Boost)
    if(Boost_FOUND)
        include_directories( ${Boost_INCLUDE_DIRS} )
        add_definitions( -DWITH_BOOST )
    else()
        message(FATAL_ERROR "\n\nERROR: You asked for Boost::uBLAS but it has not been found.\n" )
    endif()
elseif(EDO_USE_LIB STREQUAL "Eigen3")
    find_package(Eigen3)
    if(EIGEN3_FOUND)
        include_directories( ${EIGEN3_INCLUDE_DIR} )
        add_definitions( -DWITH_EIGEN )
    else()
        message(FATAL_ERROR  "\n\nERROR: You asked for Eigen3 but it has not been found.\n" )
    endif()

else()
    # FIXME ideally, we would have a minimal implementation with STL vectors…
    message(FATAL_ERROR  "\n\nYou must set EDO_USE_LIB to either 'uBLAS' or 'Eigen3'.\n" )
endif()


######################################################################################
### Include subdirectories
######################################################################################

add_subdirectory(doc)
add_subdirectory(src)

if(ENABLE_CMAKE_TESTING AND EIGEN3_FOUND)  # see edoNormalAdaptive
    add_subdirectory(test)
else()
    if( NOT EIGEN3_FOUND )
        message("Eigen3 has not been found, cannot build EDO tests")
    endif()
endif()

if(ENABLE_CMAKE_EXAMPLE)
    if(${CMAKE_VERBOSE_MAKEFILE})
        message("EDO examples:")
    endif(${CMAKE_VERBOSE_MAKEFILE})
    add_subdirectory(application)
endif()

