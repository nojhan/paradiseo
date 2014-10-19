##########################################################################################
### EO Doc generation using Doxygen
##########################################################################################
IF (DOXYGEN_FOUND)

  SET(DOC_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH "EO documentation directory")
  SET(EO_DOC_CONFIG_FILE "eo.cfg" CACHE PATH "EO documentation configuration file")

  # Copy necessary doc files
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/index.h ${DOC_DIR}/index.h COPYONLY)
  FILE(GLOB header_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.htm*)
  FILE(GLOB pdf_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.pdf)
  FILE(GLOB jpg_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.jpg)
  FOREACH (file ${header_files} ${pdf_files} ${jpg_files})
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/${file} ${DOC_DIR}/${file} COPYONLY)
  ENDFOREACH (file)

  # define the doc target
  IF (DOXYGEN_EXECUTABLE)
  # Creating the custom target
    if(UNIX AND NOT ${CMAKE_VERBOSE_MAKEFILE})
        add_custom_target(doc-eo
            COMMAND ${DOXYGEN_EXECUTABLE} ${EO_DOC_CONFIG_FILE} 2> /dev/null > /dev/null
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
    else(UNIX AND NOT ${CMAKE_VERBOSE_MAKEFILE})
        ADD_CUSTOM_TARGET(doc-eo
            COMMAND ${DOXYGEN_EXECUTABLE} ${EO_DOC_CONFIG_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
      endif()
  ENDIF (DOXYGEN_EXECUTABLE)

  # configure cfg file
  CONFIGURE_FILE(
    "${CMAKE_CURRENT_SOURCE_DIR}/${EO_DOC_CONFIG_FILE}.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${EO_DOC_CONFIG_FILE}"
    )

  INSTALL(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DESTINATION local/share/${PROJECT_TAG}/eo COMPONENT doc
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "cmake_install.cmake" EXCLUDE
    PATTERN "Makefile" EXCLUDE
    PATTERN "eo.cfg" EXCLUDE
    PATTERN "eo.doxytag" EXCLUDE
    )
ELSE (DOXYGEN_FOUND)
  MESSAGE(STATUS "Unable to generate the documentation, Doxygen package not found")
ENDIF (DOXYGEN_FOUND)
