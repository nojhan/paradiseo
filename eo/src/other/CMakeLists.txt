FILE(GLOB HDRS *.h external_eo)
INSTALL(FILES ${HDRS} DESTINATION include/eo/other COMPONENT headers)
