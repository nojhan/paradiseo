######################################################################################
### 0) Documentation
######################################################################################


if(DOXYGEN_FOUND)
    # Directory where the generation will be launched
    set(EDO_DOC_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "Documentation directory" FORCE)
    # Name of the doxygene configuration file
    set(EDO_DOC_CONFIG_FILE "edo.doxyfile" CACHE INTERNAL "Documentation configuration file")
    if(DOXYGEN_EXECUTABLE)
        # Creating the custom target
        if(UNIX AND NOT ${CMAKE_VERBOSE_MAKEFILE})
            add_custom_target(doc-edo
                COMMAND ${DOXYGEN_EXECUTABLE} ${EDO_DOC_CONFIG_FILE} 2> /dev/null > /dev/null
                              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                              )
        else(UNIX AND NOT ${CMAKE_VERBOSE_MAKEFILE})
            add_custom_target(doc-edo
                COMMAND ${DOXYGEN_EXECUTABLE} ${EDO_DOC_CONFIG_FILE}
                              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                              )
        endif(UNIX AND NOT ${CMAKE_VERBOSE_MAKEFILE})
    endif(DOXYGEN_EXECUTABLE)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${EDO_DOC_CONFIG_FILE}.cmake"
        "${EDO_DOC_DIR}/${EDO_DOC_CONFIG_FILE}")
    install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DESTINATION local/share${INSTALL_SUB_DIR}/edo COMPONENT doc
        PATTERN "CMakeFiles" EXCLUDE
        PATTERN "cmake_install.cmake" EXCLUDE
        PATTERN "CTestTestfile.cmake" EXCLUDE
        PATTERN "Makefile" EXCLUDE
        PATTERN "edo.cfg" EXCLUDE
        PATTERN "edo.doxytag" EXCLUDE
    )
else(DOXYGEN_FOUND)
    message(STATUS "Unable to generate the documentation, Doxygen package not found")
endif(DOXYGEN_FOUND)

