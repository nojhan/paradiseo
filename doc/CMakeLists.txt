## EO Module
add_subdirectory(eo)

if(NOT EO_ONLY)

    ## MO Module
    if(MO)
        add_subdirectory(mo)
    endif(MO)

    ## EDO Module
    if(EDO)
        add_subdirectory(edo)
    endif(EDO)

    ## MOEO Module
    if(MOEO)
        add_subdirectory(moeo)
    endif(MOEO)

    ## SMP Module
    if(SMP)
        add_subdirectory(smp)
    endif(SMP)

    ## EOMPI Module
    #if(EOMPI)
    #    add_subdirectory(eompi)
    #endif(EOMPI)

    ## EOSERIAL Module
    #if(EOSERIAL)
    #    add_subdirectory(eoserial)
    #endif(EOSERIAL)

endif()