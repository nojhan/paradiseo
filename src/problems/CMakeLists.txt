######################################################################################
### 1) Look for directory and install headers
######################################################################################

install(DIRECTORY eval
        DESTINATION  ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/problems
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h"
        )
