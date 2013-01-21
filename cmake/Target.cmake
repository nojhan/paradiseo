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
    if(SMP)
        add_custom_target(doc
            COMMAND make doc-eo
            COMMAND make doc-edo
            COMMAND make doc-mo
            COMMAND make doc-moeo
            COMMAND make doc-smp
        )
    else()
        add_custom_target(doc
            COMMAND make doc-eo
            COMMAND make doc-edo
            COMMAND make doc-mo
            COMMAND make doc-moeo
        )
    endif()
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

