######################################################################################
### 0) Set up components
######################################################################################

set(CPACK_COMPONENTS_ALL 
    libraries
    headers
    tests
    examples
    doc
)
set(CPACK_ALL_INSTALL_TYPES Minimal Full)

######################################################################################
### 1) Describing components
######################################################################################

set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Libraries")
set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION "ParadisEO Libraries : EO, MO, MOEO")
set(CPACK_COMPONENT_LIBRARIES_INSTALL_TYPES Minimal Full)
set(CPACK_COMPONENT_LIBRARIES_REQUIRED)

set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "Headers")
set(CPACK_COMPONENT_HEADERS_DESCRIPTION "C++ headers for using ParadisEO")
set(CPACK_COMPONENT_HEADERS_DEPENDS libraries)
set(CPACK_COMPONENT_HEADERS_INSTALL_TYPES Minimal Full)
set(CPACK_COMPONENT_LIBRARIES_REQUIRED)

set(CPACK_COMPONENT_TESTS_DISPLAY_NAME "Tests")
set(CPACK_COMPONENT_TESTS_DESCRIPTION "Tests")
set(CPACK_COMPONENT_TESTS_DEPENDS libraries)
set(CPACK_COMPONENT_TESTS_INSTALL_TYPES Full)

set(CPACK_COMPONENT_EXAMPLES_DISPLAY_NAME "Examples")
set(CPACK_COMPONENT_EXAMPLES_DESCRIPTION "Examples and lessons")
set(CPACK_COMPONENT_EXAMPLES_DEPENDS libraries)
set(CPACK_COMPONENT_EXAMPLES_INSTALL_TYPES Full)

set(CPACK_COMPONENT_DOC_DISPLAY_NAME "Documentation")
set(CPACK_COMPONENT_DOC_DESCRIPTION "ParadisEO documentation")
set(CPACK_COMPONENT_DOC_INSTALL_TYPES Full)

######################################################################################
### 2) Set up general information about packaging
######################################################################################

# For more details: http://www.cmake.org/Wiki/CMake:Component_Install_With_CPack

#cpack package information
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README")
set(CPACK_PACKAGE_DESCRIPTION "ParadisEO is a white-box object-oriented framework dedicated to the flexible design of metaheuristics. This template-based, ANSI-C++ compliant computation library is portable across both Windows system and sequential platforms (Unix, Linux, Mac OS X, etc.). ParadisEO is distributed under the CeCill license and can be used under several environments.")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A Software Framework for Metaheuristics")
set(CPACK_PACKAGE_VENDOR "Inria/Thales")
set(CPACK_PACKAGE_CONTACT "paradiseo-help@lists.gforge.inria.fr")
set(CPACK_PACKAGE_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})
set(CPACK_STRIP_FILES ${PACKAGE_NAME})
set(CPACK_SOURCE_STRIP_FILES "bin/${PROJECT_NAME}")
set(CPACK_PACKAGE_EXECUTABLES "${PROJECT_NAME}" "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION_MAJOR "${VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VERSION_PATCH}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${PROJECT_NAME}")
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}-${ARCH}")

if(UNIX)
    if(APPLE)
        # Generators for Mac OS X
        set(CPACK_BINARY_PACKAGEMARKER "ON")
        set(CPACK_BINARY_TGZ "OFF")
        set(CPACK_BINARY_STGZ "OFF")
        
        #set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/mac.rtf")

    else()
        # Generators for Unix-like
        set(CPACK_GENERATOR "DEB;RPM")
        # Determine architecture
        include(CheckTypeSize)
        check_type_size(void* SIZEOF_VOID_PTR)
        if("${SIZEOF_VOID_PTR}" STREQUAL "4")
            set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE i386)
        elseif("${SIZEOF_VOID_PTR}" STREQUAL "8")
            set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
        else()
            message(FATAL_ERROR "Unsupported architecture")
            return()
        endif()

    endif()
else(UNIX)

    # Generator for Windows
    set(CPACK_GENERATOR "NSIS")
    #set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/winicon.bpm")
  
endif()

######################################################################################
### 3) And finally, include cpack, this is the last thing to do.
######################################################################################

include(CPack)
