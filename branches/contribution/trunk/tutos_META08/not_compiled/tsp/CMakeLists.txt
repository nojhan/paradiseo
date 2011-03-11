
######################################################################################
### 1) Where must cmake go now ?
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS  ${SOURCES_DIR}/tsp/benchs)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory 
      ${SOURCES_DIR}/tsp/benchs
      ${INSTALL_DIR}/tsp/benchs)

SUBDIRS(src)

######################################################################################

