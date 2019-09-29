# The script use the following variables as search paths, if they are defined:
# - PARADISEO_ROOT : the project root
# - PARADISEO_DIR : the build/install directory with libraries binaries
#
# The following variables are filled out:
# - PARADISEO_INCLUDE_DIR : EO, MO and MOEO source dir
# - EO_INCLUDE_DIR :        EO source dir
# - EDO_INCLUDE_DIR :        EO source dir
# - MO_INCLUDE_DIR :        MO source dir
# - MOEO_INCLUDE_DIR :      MOEO source dir. WARNING : You have ton include MO before !
# - PARADISEO_LIBRARIES :   the list of all required modules
# - PARADISEO_XXX_LIBRARY : the name of the library to link for the required module   
# - PARADISEO_XXX_FOUND :   true if the required module is found
# - PARADISEO_FOUND :       true if all required modules are found
#
# Here are the components:
# - eo
# - edo
# - PyEO
# - es
# - ga
# - cma
# - flowshop
# - moeo
# - smp
# - peo
# You can use find_package(Paradiseo COMPONENTS ... ) to enable one or several components. If you not specifie component, all components will be load except SMP for compatibility reasons.
#
# Output
# ------
#
# example:
#   find_package(Paradiseo COMPONENTS eo eoutils cma es flowshop ga moeo REQUIRED)
#   include_directories(${PARADISEO_INCLUDE_DIR})
#   add_executable(example ...)
#   target_link_libraries(examplep ${PARADISEO_LIBRARIES})

if(UNIX)
    set(INSTALL_SUB_DIR /paradiseo)
endif()

if(PARADISEO_DIR)
    # CMake config module is case sensitive
    set(Paradiseo_DIR ${PARADISEO_DIR})
endif()

# enabled components
if (Paradiseo_FIND_COMPONENTS STREQUAL "")
    set(PARADISEO_LIBRARIES_TO_FIND eo eoutils cma es flowshop ga moeo)
else()
    set(PARADISEO_LIBRARIES_TO_FIND ${Paradiseo_FIND_COMPONENTS})
endif()
message(STATUS "${PARADISEO_LIBRARIES_TO_FIND}")

#set the build directory
#set(BUILD_DIR build)

# Path
set(PARADISEO_SRC_PATHS
        ${PARADISEO_ROOT}
        $ENV{PARADISEO_ROOT}
        /usr/local/
        /usr/
        /sw # Fink
        /opt/local/ # DarwinPorts
        /opt/csw/ # Blastwave
        /opt/
        [KEY_CURRENT_USER\\Software\\Inria\\ParadisEO]/local
        [HKEY_LOCAL_MACHINE\\Software\\Inria\\ParadisEO]/local
)

find_path(EO_INCLUDE_DIR eo
          PATH_SUFFIXES include${INSTALL_SUB_DIR}/eo eo/src
          PATHS ${PARADISEO_SRC_PATHS})

find_path(MO_INCLUDE_DIR mo
          PATH_SUFFIXES include${INSTALL_SUB_DIR}/mo mo/src
          PATHS ${PARADISEO_SRC_PATHS})

find_path(MOEO_INCLUDE_DIR moeo
          PATH_SUFFIXES include${INSTALL_SUB_DIR}/moeo moeo/src
          PATHS ${PARADISEO_SRC_PATHS})

set(PARADISEO_INCLUDE_DIR ${EO_INCLUDE_DIR} ${MO_INCLUDE_DIR} ${MOEO_INCLUDE_DIR})

# Specific for SMP, EDO and PEO
foreach(COMP ${PARADISEO_LIBRARIES_TO_FIND})
    if(${COMP} STREQUAL "smp")
        set(SMP_FOUND true)
        find_path(SMP_INCLUDE_DIR smp
              PATH_SUFFIXES include${INSTALL_SUB_DIR}/smp smp/src
              PATHS ${PARADISEO_SRC_PATHS})
    elseif(${COMP} STREQUAL "edo")
        set(EDO_FOUND true)
        find_path(EDO_INCLUDE_DIR edo
          PATH_SUFFIXES include${INSTALL_SUB_DIR}/edo edo/src
          PATHS ${PARADISEO_SRC_PATHS})
    elseif(${COMP} STREQUAL "peo")
        set(PEO_FOUND true)
        find_path(PEO_INCLUDE_DIR peo
              PATH_SUFFIXES include${INSTALL_SUB_DIR}/peo peo/src
              PATHS ${PARADISEO_SRC_PATHS})
    endif()
endforeach()

if(SMP_FOUND)
    set(PARADISEO_INCLUDE_DIR ${PARADISEO_INCLUDE_DIR} ${SMP_INCLUDE_DIR})
endif()

if(EDO_FOUND)
    set(PARADISEO_INCLUDE_DIR ${PARADISEO_INCLUDE_DIR} ${EDO_INCLUDE_DIR})
endif()

if(PEO_FOUND)
    set(PARADISEO_INCLUDE_DIR ${PARADISEO_INCLUDE_DIR} ${PEO_INCLUDE_DIR})
endif()

# find the requested modules
set(PARADISEO_FOUND true) # will be set to false if one of the required modules is not found

set(FIND_PARADISEO_LIB_PATHS
        # ${PARADISEO_ROOT}/${BUILD_DIR}
        ${Paradiseo_DIR}
        $ENV{PARADISEO_ROOT}/build
        $ENV{PARADISEO_ROOT}/release
        $ENV{PARADISEO_ROOT}/debug
        ${PARADISEO_ROOT}/build
        ${PARADISEO_ROOT}/release
        ${PARADISEO_ROOT}/debug
        /usr/local/
        /usr/
        /sw # Fink
        /opt/local/ # DarwinPorts
        /opt/csw/ # Blastwave
        /opt/
        [KEY_CURRENT_USER\\Software\\Inria\\ParadisEO]/local
        [HKEY_LOCAL_MACHINE\\Software\\Inria\\ParadisEO]/local
)

#Suffixes
set(PARADISEO_LIB_PATHS_SUFFIXES
        eo/lib 
        edo/lib 
        mo/lib 
        moeo/lib 
        moeo/tutorial/examples/flowshop/lib #For flowshop library
        smp/lib
        peo/lib
        lib 
        lib32 
        lib64
        )

foreach(FIND_PARADISEO_COMPONENT ${PARADISEO_LIBRARIES_TO_FIND})
    string(TOUPPER ${FIND_PARADISEO_COMPONENT} FIND_PARADISEO_COMPONENT_UPPER)
    # release library
    find_library(PARADISEO_${FIND_PARADISEO_COMPONENT_UPPER}_LIBRARY
                 NAMES ${FIND_PARADISEO_COMPONENT}
                 PATH_SUFFIXES ${PARADISEO_LIB_PATHS_SUFFIXES}
                 PATHS ${FIND_PARADISEO_LIB_PATHS})
    if (PARADISEO_${FIND_PARADISEO_COMPONENT_UPPER}_LIBRARY)
        # library found
        set(PARADISEO_${FIND_PARADISEO_COMPONENT_UPPER}_FOUND true)
    else()
        # library not found
        set(PARADISEO_FOUND false)
        set(PARADISEO_${FIND_PARADISEO_COMPONENT_UPPER}_FOUND false)
        set(FIND_PARADISEO_MISSING "${FIND_PARADISEO_MISSING} ${FIND_PARADISEO_COMPONENT}")
    endif()
    set(PARADISEO_LIBRARIES ${PARADISEO_LIBRARIES} "${PARADISEO_${FIND_PARADISEO_COMPONENT_UPPER}_LIBRARY}")
endforeach()

# handle result
if(PARADISEO_FOUND)
    message(STATUS "Found the following ParadisEO include directories:")
    message(STATUS "\tEO\t: " ${EO_INCLUDE_DIR})
    message(STATUS "\tMO\t: " ${MO_INCLUDE_DIR})
    message(STATUS "\tMOEO\t: " ${MOEO_INCLUDE_DIR})
    if(SMP_FOUND)
        message(STATUS "\tSMP\t: " ${SMP_INCLUDE_DIR})
    endif()
    if(EDO_FOUND)
        message(STATUS "\tEDO\t: " ${EDO_INCLUDE_DIR})
    endif()
    if(PEO_FOUND)
        message(STATUS "\tPEO\t: " ${PEO_INCLUDE_DIR})
    endif()
else()
    # include directory or library not found
    message(FATAL_ERROR "Could NOT find ParadisEO (missing \t: ${FIND_PARADISEO_MISSING})")
endif()
