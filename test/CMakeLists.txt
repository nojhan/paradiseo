## EO Module
add_subdirectory(${EO_TEST_DIR})

if(NOT EO_ONLY)

    ## MO Module
    if(MO)
        add_subdirectory(${MO_TEST_DIR})
    endif(MO)

    ## EDO Module
    if(EDO)
        if(EDO_USE_LIB STREQUAL "Eigen3")
            find_package(Eigen3)
            if(EIGEN3_FOUND)  # see edoNormalAdaptive
                add_subdirectory(${EDO_TEST_DIR})
            else(EIGEN3_FOUND)
                message(FATAL_ERROR  "\n\nERROR: You asked for Eigen3 but it has not been found.\n" )
            endif(EIGEN3_FOUND)
        elseif(EDO_USE_LIB STREQUAL "uBLAS")
            find_package(Boost)
            if(Boost_FOUND)
                message("Warning: EDO_USE_LIB MATCHES ${EDO_USE_LIB} but Eigen3 recommended for tests")
                add_subdirectory(${EDO_TEST_DIR})
            else(Boost_FOUND)
                message(FATAL_ERROR "\n\nERROR: You asked for Boost::uBLAS but it has not been found.\n" )
            endif(Boost_FOUND)
        else()
            message(FATAL_ERROR  "\n\nYou must set EDO_USE_LIB to either 'uBLAS' or 'Eigen3'.\n" )
        endif()
    endif(EDO)

    ## MOEO Module
    if(MOEO)
        add_subdirectory(${MOEO_TEST_DIR})
    endif(MOEO)

    ## SMP Module
    if(SMP)
        add_subdirectory(${SMP_TEST_DIR})
    endif(SMP)

    ## EOMPI Module
    if(EOMPI)
        add_subdirectory(${EOMPI_TEST_DIR})
    endif(EOMPI)

    ## EOSERIAL Module
    #if(EOSERIAL)
    #    add_subdirectory(${EOSERIAL_TEST_DIR})
    #endif(EOSERIAL)

endif()