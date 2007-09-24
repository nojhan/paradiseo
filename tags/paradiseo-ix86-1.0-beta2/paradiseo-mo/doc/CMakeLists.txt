
### ParadisEO-MO Doc generation using Doxygen ###

# Find the Doxygen package - will set all the variables required to use Doxygen
FIND_PACKAGE(Doxygen)

IF (DOXYGEN_FOUND)

SET(MO_DOC_COMMENT "ParadisEO-MO framework documentation")
SET(DOXYGEN_INPUT "mo.doxyfile")
SET(DOXYGEN_OUTPUT "ParadisEO-MO documentation")

ADD_CUSTOM_COMMAND(
  OUTPUT ${DOXYGEN_OUTPUT}
  COMMAND ${CMAKE_COMMAND} -E echo_append "Building documentation from ${MO_DOC_DIR} ..."
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
  COMMAND ${CMAKE_COMMAND} -E echo "Done."
  WORKING_DIRECTORY ${MO_DOC_DIR}
  COMMENT ${MO_DOC_COMMENT}
  DEPENDS ${DOXYGEN_INPUT}
  )

# The main target launched with "make doc"
ADD_CUSTOM_TARGET(doc DEPENDS ${DOXYGEN_OUTPUT})

# A forced additionnal target that generate the documentation in the current directory
ADD_CUSTOM_TARGET(doc_forced
  COMMAND ${CMAKE_COMMAND} -E echo_append "Building documentation from current directory ..."
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
  COMMAND ${CMAKE_COMMAND} -E echo " Done."
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

ELSE (DOXYGEN_FOUND)
	MESSAGE(STATUS "Unable to generate the document, Doxygen package not found")
ENDIF (DOXYGEN_FOUND)



