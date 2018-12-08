######################################################################################
### cleanall will delete all files and folders in build directory
######################################################################################

if(UNIX)
    add_custom_target(cleanall COMMAND cd ${CMAKE_BINARY_DIR} && rm -rf *)
endif(UNIX)

######################################################################################
### Doc-all enable to build all documentations in one target
######################################################################################

if(DOXYGEN_FOUND AND DOXYGEN_EXECUTABLE)
    # FIXME this would work in cmake 3.13
    # set(DOC_EO "make doc-eo")
    # if(NOT EO_ONLY)
    #     set(DOC_MO "make doc-mo")
    #     set(DOC_MOEO "make doc-moeo")
    #     if(EDO)
    #         set(DOC_EDO "make doc-edo")
    #     else()
    #         set(DOC_EDO "")
    #     endif()
    #     if(SMP)
    #         set(DOC_SMP "make doc-smp")
    #     else()
    #         set(DOC_SMP "")
    #     endif()
    #     if(MPI)
    #         set(DOC_MPI "make doc-mpi")
    #     else()
    #         set(DOC_MPI "")
    #     endif()
    # endif()
    #
    # add_custom_target(doc
    #     COMMAND ${DOC_EO}
    #     COMMAND ${DOC_MO}
    #     COMMAND ${DOC_MOEO}
    #     COMMAND ${DOC_EDO}
    #     COMMAND ${DOC_SMP}
    #     COMMAND ${DOC_MPI}
    # )
    # FIXME in the meantime, we must enumerate...
    if(EO_ONLY)
        add_custom_target(doc
            COMMAND make doc-eo
        )
    else()
        # No optional module.
        if(NOT EDO AND NOT SMP AND NOT MPI)
            add_custom_target(doc
                COMMAND make doc-eo
                COMMAND make doc-mo
                COMMAND make doc-moeo
            )
        endif()

        # One optional module.
        if(EDO AND NOT SMP AND NOT MPI)
            add_custom_target(doc
                COMMAND make doc-eo
                COMMAND make doc-mo
                COMMAND make doc-moeo
                COMMAND make doc-edo
            )
        endif()
        if(NOT EDO AND SMP AND NOT MPI)
            add_custom_target(doc
                COMMAND make doc-eo
                COMMAND make doc-mo
                COMMAND make doc-moeo
                COMMAND make doc-smp
            )
        endif()
        if(NOT EDO AND NOT SMP AND MPI)
            add_custom_target(doc
                COMMAND make doc-eo
                COMMAND make doc-mo
                COMMAND make doc-moeo
                COMMAND make doc-mpi
            )
        endif()

        # Two optional modules.
        if(NOT EDO AND SMP AND MPI)
            add_custom_target(doc
                COMMAND make doc-eo
                COMMAND make doc-mo
                COMMAND make doc-moeo
                COMMAND make doc-smp
                COMMAND make doc-mpi
            )
        endif()
        if(EDO AND NOT SMP AND MPI)
            add_custom_target(doc
                COMMAND make doc-eo
                COMMAND make doc-mo
                COMMAND make doc-moeo
                COMMAND make doc-edo
                COMMAND make doc-mpi
            )
        endif()
        if(EDO AND SMP AND NOT MPI)
            add_custom_target(doc
                COMMAND make doc-eo
                COMMAND make doc-mo
                COMMAND make doc-moeo
                COMMAND make doc-edo
                COMMAND make doc-smp
            )
        endif()

        # Three optional modules
        if(EDO AND SMP AND MPI)
            add_custom_target(doc
                COMMAND make doc-eo
                COMMAND make doc-mo
                COMMAND make doc-moeo
                COMMAND make doc-edo
                COMMAND make doc-smp
                COMMAND make doc-mpi
            )
        endif()

    endif(EO_ONLY)
endif(DOXYGEN_FOUND AND DOXYGEN_EXECUTABLE)

######################################################################################
### Perform covering test if lcov is found
######################################################################################

if(PROFILING)
    find_program(LCOV 
        NAMES lcov
        PATHS
        "/usr/local/bin /usr/bin [HKEY_LOCAL_MACHINE\\SOFTWARE\\Rational Software\\Purify\\Setup;InstallFolder] [HKEY_CURRENT_USER\\Software]"
        DOC "Path to the memory checking command, used for memory error detection.")
    if(LCOV)
        add_custom_target(coverage
            COMMAND make
            COMMAND ctest
            COMMAND lcov -d . -c -o output.info
            COMMAND lcov -r output.info '*/tutorial*' -o output.info
            COMMAND lcov -r output.info '/usr*' -o output.info
            COMMAND lcov -r output.info '*/test*' -o output.info
            COMMAND lcov -r output.info '*/eo*' -o output.info
            COMMAND lcov -r output.info '*/mo*' -o output.info
            COMMAND lcov -r output.info '*/moeo*' -o output.info
            COMMAND lcov -r output.info '*/problems*' -o output.info
            COMMAND genhtml output.info -o coverage/ --highlight --legend
        )
    else(LCOV)
        message(STATUS "Could NOT find Lcov")
    endif(LCOV)
endif(PROFILING)

