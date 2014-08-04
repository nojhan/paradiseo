######################################################################################
### 0) Look for headers
######################################################################################

install(FILES mo.h DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG} COMPONENT headers)

######################################################################################
### 1) Install directories
######################################################################################

install(DIRECTORY acceptCrit algo comparator continuator coolingSchedule eval explorer memory neighborhood perturb problems sampling
        DESTINATION  ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/mo 
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h"
        )

