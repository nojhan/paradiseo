######################################################################################
### 0) Documentation
######################################################################################

if(DOXYGEN_FOUND)
    # Directory where the generation will be launched
    set(SMP_DOC_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "Documentation directory" FORCE)
    # Name of the doxygene configuration file
    set(SMP_DOC_CONFIG_FILE "smp.doxyfile" CACHE INTERNAL "Documentation configuration file")
    if(DOXYGEN_EXECUTABLE)
        # Creating the custom target
        if(UNIX AND NOT ${CMAKE_VERBOSE_MAKEFILE})
            add_custom_target(doc-smp
                COMMAND ${DOXYGEN_EXECUTABLE} ${SMP_DOC_CONFIG_FILE} 
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        else(UNIX AND NOT ${CMAKE_VERBOSE_MAKEFILE})
            add_custom_target(doc-smp
                COMMAND ${DOXYGEN_EXECUTABLE} ${SMP_DOC_CONFIG_FILE}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        endif(UNIX AND NOT ${CMAKE_VERBOSE_MAKEFILE})
    endif(DOXYGEN_EXECUTABLE)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${SMP_DOC_CONFIG_FILE}.cmake"
                   "${SMP_DOC_DIR}/${SMP_DOC_CONFIG_FILE}")
    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DESTINATION share${INSTALL_SUB_DIR}/smp COMPONENT doc
        PATTERN "CMakeFiles" EXCLUDE
        PATTERN "cmake_install.cmake" EXCLUDE
        PATTERN "CTestTestfile.cmake" EXCLUDE
        PATTERN "Makefile" EXCLUDE
        PATTERN "smp.cfg" EXCLUDE
        PATTERN "smp.doxytag" EXCLUDE
    )
else(DOXYGEN_FOUND)
    message(STATUS "Unable to generate the documentation, Doxygen package not found")
endif(DOXYGEN_FOUND)

