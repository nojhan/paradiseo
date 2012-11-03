######################################################################################
### 0) Where must cmake go now ?
######################################################################################

add_subdirectory(core)
add_subdirectory(rmc)

######################################################################################
### 3) Look for headers
######################################################################################

file(GLOB HDRS peo)
install(FILES ${HDRS} DESTINATION include${INSTALL_SUB_DIR}/peo COMPONENT headers)

######################################################################################
### 2) Install directories
######################################################################################

install(DIRECTORY core rmc
        DESTINATION include${INSTALL_SUB_DIR}/peo
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h"
        )

