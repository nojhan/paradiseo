
IF(EDO_USE_LIB STREQUAL "uBLAS")
    FIND_PACKAGE(Boost )
    IF( Boost_FOUND )
        INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
        ADD_DEFINITIONS( -DWITH_BOOST )
    ELSE()
        MESSAGE( "ERROR: You asked for Boost::uBLAS but it has not been found." )
        SET(IS_FATAL 1)
    ENDIF()
ELSEIF( EDO_USE_LIB STREQUAL "Eigen3" )
    # FIXME FindEigen3.cmake does not work
    #find_package(Eigen3)
    #include_directories(EIGEN3_INCLUDE_DIR)
    SET( EIGEN3_FOUND 1)
    SET( EIGEN3_INCLUDE_DIR "/usr/include/eigen3/" )

    IF( EIGEN3_FOUND )
        INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIR} )
        ADD_DEFINITIONS( -DWITH_EIGEN )
    ELSE()
        MESSAGE( "ERROR: You asked for Eigen3 but it has not been found." )
        SET(IS_FATAL 1)
    ENDIF()

ELSE()
    # FIXME ideally, we would have a minimal implementation with STL vectors…
    MESSAGE( "You must set EDO_USE_LIB to either 'uBLAS' or 'Eigen3'." )
    SET(IS_FATAL 1)
ENDIF()


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
        message("EDO examples:")
    endif(${CMAKE_VERBOSE_MAKEFILE})
    add_subdirectory(application)
endif(ENABLE_CMAKE_EXAMPLE)

