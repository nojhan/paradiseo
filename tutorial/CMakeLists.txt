## EO Module
if(${CMAKE_VERBOSE_MAKEFILE})
    message("EO examples:")
endif(${CMAKE_VERBOSE_MAKEFILE})
add_subdirectory(${EO_TUTORIAL_DIR})

if(NOT EO_ONLY)

    ## MO Module
    if(MO)
        if(${CMAKE_VERBOSE_MAKEFILE})
            message("MO examples:")
        endif(${CMAKE_VERBOSE_MAKEFILE})
        add_subdirectory(${MO_TUTORIAL_DIR})
    endif(MO)

    ## EDO Module
    if(EDO)
        if(${CMAKE_VERBOSE_MAKEFILE})
            message("EDO examples:")
        endif(${CMAKE_VERBOSE_MAKEFILE})
        add_subdirectory(${EDO_TUTORIAL_DIR})

    endif(EDO)

    ## MOEO Module
    if(MOEO)
        if(${CMAKE_VERBOSE_MAKEFILE})
            message("MOEO examples:")
        endif(${CMAKE_VERBOSE_MAKEFILE})
        add_subdirectory(${MOEO_TUTORIAL_DIR})
    endif(MOEO)

    ## SMP Module
    if(SMP)
        if(${CMAKE_VERBOSE_MAKEFILE})
            message("SMP examples:")
        endif(${CMAKE_VERBOSE_MAKEFILE})
        add_subdirectory(${SMP_TUTORIAL_DIR})
    endif(SMP)

    ## EOMPI Module
    #if(EOMPI)
    #    if(${CMAKE_VERBOSE_MAKEFILE})
    #        message("EOMPI examples:")
    #    endif(${CMAKE_VERBOSE_MAKEFILE})
    #    add_subdirectory(${EOMPI_TUTORIAL_DIR})
    #endif(EOMPI)

    ## EOSERIAL Module
    #if(EOSERIAL)
    #    if(${CMAKE_VERBOSE_MAKEFILE})
    #        message("EOSERIAL examples:")
    #    endif(${CMAKE_VERBOSE_MAKEFILE})
    #    add_subdirectory(${EOSERIAL_TUTORIAL_DIR})
    #endif(EOSERIAL)

endif()