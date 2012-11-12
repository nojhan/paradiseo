######################################################################################
### 0) Look for headers
######################################################################################

file(GLOB HDRS *.h mo)
install(FILES ${HDRS} DESTINATION include${INSTALL_SUB_DIR}/mo COMPONENT headers)

######################################################################################
### 1) Install directories
######################################################################################

install(DIRECTORY acceptCrit algo comparator continuator coolingSchedule eval explorer memory neighborhood perturb problems sampling
        DESTINATION include${INSTALL_SUB_DIR}/mo 
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h"
        )

