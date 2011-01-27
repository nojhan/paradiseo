#######################################################################################
### Doc generation using Doxygen
#######################################################################################

IF (DOXYGEN_FOUND)
  SET(DOC_DIR ${CMAKE_BINARY_DIR}/doc CACHE PATH "documentation directory")
  SET(DOC_CONFIG_FILE "doxyfile" CACHE PATH "documentation configuration file")

  # define the doc target
  IF (DOXYGEN_EXECUTABLE)
    ADD_CUSTOM_TARGET(doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOC_CONFIG_FILE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )
  ENDIF (DOXYGEN_EXECUTABLE)

  # configure doxyfile file
  CONFIGURE_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/${DOC_CONFIG_FILE}.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${DOC_CONFIG_FILE}"
    )

  INSTALL(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DESTINATION share/edo COMPONENT libraries
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "cmake_install.cmake" EXCLUDE
    PATTERN "Makefile" EXCLUDE
    PATTERN "doxyfile" EXCLUDE
    )
ELSE (DOXYGEN_FOUND)
  MESSAGE(STATUS "Unable to generate the documentation, Doxygen package not found")
ENDIF (DOXYGEN_FOUND)

#######################################################################################
